"""General-purpose phase-channel hierarchical encoder, generation 2.

Task heads and pretraining objectives are attached separately from this backbone.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .phasor_hierarchical import Block, positional, physical_features
from osc_tools.ml.layers.transformer_blocks import FastKANLayer


@dataclass(frozen=True)
class HierarchicalV2Config:
    architecture_id: str = 'phasor_hierarchical_v2'
    harmonics: int = 13
    channel_width: int = 48
    channel_heads: int = 3
    width: int = 128
    heads: int = 4
    channel_layers: int = 4
    local_layers: int = 12
    global_layers: int = 12
    fusion_layers: int = 4
    block_tokens: int = 80
    block_stride: int = 40
    grids: int = 5
    dropout: float = .05
    drop_path_rate: float = .1
    checkpoint_blocks: bool = True
    current_noise: float = 1/2000
    voltage_noise: float = 1/300


class PhasorHierarchicalKANV2(nn.Module):
    def __init__(self, cfg=HierarchicalV2Config()):
        super().__init__(); self.cfg = cfg
        h, c, d = cfg.harmonics, cfg.channel_width, cfg.width
        if cfg.current_noise<=0 or cfg.voltage_noise<=0: raise ValueError('Noise limits must be positive')
        if c % cfg.channel_heads or d % cfg.heads or c % 2 or d % 2:
            raise ValueError('Invalid attention widths')
        if not 0 < cfg.block_stride <= cfg.block_tokens or not 0 <= cfg.drop_path_rate < 1:
            raise ValueError('Invalid overlap/drop path')
        self.pair = nn.Sequential(FastKANLayer(7,16,num_grids=cfg.grids,use_layernorm=False),
                                  FastKANLayer(16,16,num_grids=cfg.grids,use_layernorm=False))
        self.harmonic_mix = nn.Sequential(nn.Linear(h*16,64),nn.SiLU(),nn.Linear(64,c))
        self.channel_id = nn.Parameter(torch.randn(6,c)*.02)
        self.channel_blocks = nn.ModuleList(Block(c,cfg.channel_heads,cfg.grids,cfg.dropout) for _ in range(cfg.channel_layers))
        self.mixer = nn.Linear(6*c,d)
        self.raw_bypass = nn.Linear(h*6*2,d,bias=False)
        make = lambda n: nn.ModuleList(Block(d,cfg.heads,cfg.grids,cfg.dropout,h) for _ in range(n))
        self.local_blocks, self.global_blocks, self.fusion_blocks = make(cfg.local_layers),make(cfg.global_layers),make(cfg.fusion_layers)
        self.summary_type = nn.Parameter(torch.randn(2,d)*.02)
        self.context_readout = nn.Linear(2*d,d)
        self.context = nn.ModuleList(nn.Linear(d,d) for _ in self.fusion_blocks)
        self.delta = nn.Linear(d,d)
        self.gate = nn.Linear(2*d,1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,math.log(.05/.95))
        self.norm = nn.LayerNorm(d)
        self.rates = {}
        for stage in (self.channel_blocks,self.local_blocks,self.global_blocks,self.fusion_blocks):
            for i, block in enumerate(stage): self.rates[id(block)] = cfg.drop_path_rate*(i+1)/len(stage)

    def run_block(self, block, x, valid, phi=None):
        rate = self.rates[id(block)] if self.training else 0.
        if rate and torch.rand(()) < rate: return x*valid.unsqueeze(-1)
        y = (checkpoint(block,x,valid,phi,use_reentrant=False)
             if self.training and self.cfg.checkpoint_blocks else block(x,valid,phi))
        return (x+(y-x)/(1-rate))*valid.unsqueeze(-1) if rate else y

    def encode(self, z, state):
        with torch.autocast(z.device.type,enabled=False):
            visible = (state==1)|(state==2)
            safe = torch.where(visible.unsqueeze(-1),z.float(),0.)
            amp = torch.sqrt(safe.square().sum(-1)+1e-16)
            encoded = self.pair(torch.cat((safe,torch.log1p(amp).unsqueeze(-1),F.one_hot(state.long(),4).float()),-1))
            channels = self.harmonic_mix(encoded.transpose(-3,-2).flatten(-2))+self.channel_id
            phi = physical_features(safe,visible)
            # Z/Y already have guards; apply the same sensitivity to vector direction.
            confidence = ((amp[...,:3]/self.cfg.current_noise-1).clamp(0,1)*
                          (amp[...,3:]/self.cfg.voltage_noise-1).clamp(0,1))
            phi = torch.cat((phi[...,:3],phi[...,3:5]*confidence.unsqueeze(-1),phi[...,5:]),-1)
            return channels,safe,visible,phi

    @staticmethod
    def merge(windows, mask, index, length):
        b,n,k,d = windows.shape
        weights = .1+.9*(1-(torch.arange(k,device=windows.device)-(k-1)/2).abs()/max(1,(k-1)/2))
        weights = weights[None,None,:]*mask
        total = int(index[-1,-1])+1
        values = windows.float()*weights.unsqueeze(-1)
        out = torch.zeros(b,total,d,device=windows.device,dtype=torch.float32).index_add(1,index.flatten(),values.reshape(b,-1,d))
        den = weights.new_zeros(b,total).index_add(1,index.flatten(),weights.reshape(b,-1))
        return (out/den.clamp_min(1e-6).unsqueeze(-1))[:,:length]

    def forward(self, z, state, times, valid, use_global=True):
        b,t,h,_,_ = z.shape; k,s = self.cfg.block_tokens,self.cfg.block_stride
        n = max(1,math.ceil((t-k)/s)+1); total=(n-1)*s+k; pad=total-t
        z=F.pad(z,(0,0,0,0,0,0,0,pad));state=F.pad(state,(0,0,0,0,0,pad))
        times=F.pad(times,(0,pad));valid=F.pad(valid,(0,pad))
        channels,safe,visible,phi = (checkpoint(self.encode,z,state,use_reentrant=False)
            if self.training and self.cfg.checkpoint_blocks else self.encode(z,state))
        valid = valid & visible.any((-1,-2))
        index = torch.arange(n,device=z.device)[:,None]*s+torch.arange(k,device=z.device)[None,:]
        mask=valid[:,index]; channel_mask=mask.unsqueeze(-1)&visible.any(-2)[:,index]
        x=channels[:,index].permute(0,1,3,2,4).reshape(b*n*6,k,-1)
        cmask=channel_mask.permute(0,1,3,2).reshape(b*n*6,k)
        window_times=times[:,index]; local_times=window_times-window_times[:,:,:1]
        x=x+positional(local_times.reshape(b*n,k),self.cfg.channel_width)[:,None].expand(-1,6,-1,-1).reshape_as(x)
        for block in self.channel_blocks: x=self.run_block(block,x,cmask)
        x=x.reshape(b,n,6,k,-1).permute(0,1,3,2,4).flatten(-2).reshape(b*n,k,-1)
        with torch.autocast(z.device.type,enabled=False): x=self.mixer(x.float())
        x=x+self.raw_bypass(safe[:,index].reshape(b*n,k,-1))
        wm=mask.reshape(b*n,k); wp=phi[:,index].reshape(b*n,k,h,3,10)
        for block in self.local_blocks: x=self.run_block(block,x,wm,wp)
        local=x
        local_sequence=self.norm(self.merge(local.reshape(b,n,k,-1),mask,index,t))
        if not use_global:
            return dict(token_features=local_sequence,local_features=local_sequence,valid_mask=valid[:,:t],times=times[:,:t],
                        global_features=x.new_zeros(b,0,self.cfg.width),global_valid_mask=valid[:,:0],global_times=times[:,:0])
        denom=wm.sum(-1,keepdim=True).clamp_min(1)
        mean=(x*wm.unsqueeze(-1)).sum(1)/denom
        maximum=x.masked_fill(~wm.unsqueeze(-1),-torch.inf).amax(1)
        maximum=torch.where(wm.any(-1,keepdim=True),maximum,0.)
        summary=torch.stack((mean,maximum),1).reshape(b,2*n,-1)
        smask=mask.any(-1).repeat_interleave(2,dim=1)
        center=(window_times*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        summary_times=center.repeat_interleave(2,dim=1)
        summary=summary+positional(summary_times,self.cfg.width)+self.summary_type.repeat(n,1)[None]
        mean_phi=(wp*wm[...,None,None,None]).sum(1)/denom[...,None,None]
        # A coordinatewise latent maximum is not a physical phasor: do not invent its physics.
        sphi=torch.stack((mean_phi,torch.zeros_like(mean_phi)),1).reshape(b,2*n,h,3,10)
        for block in self.global_blocks: summary=self.run_block(block,summary,smask,sphi)
        context=self.context_readout(summary.reshape(b,n,-1)).reshape(b*n,1,-1).expand(-1,k,-1)
        candidate=local
        for block,project in zip(self.fusion_blocks,self.context):
            candidate=self.run_block(block,candidate+.05*project(context),wm,wp)
        gate=torch.sigmoid(self.gate(torch.cat((local,context),-1)))
        fused=local+gate*torch.tanh(self.delta(candidate-local))
        sequence=self.norm(self.merge(fused.reshape(b,n,k,-1),mask,index,t))
        return dict(token_features=sequence,local_features=local_sequence,global_features=summary,
                    valid_mask=valid[:,:t],global_valid_mask=smask,times=times[:,:t],global_times=summary_times,
                    context_gate=self.merge(gate.reshape(b,n,k,1),mask,index,t))
