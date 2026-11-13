"""Third-generation phase-channel encoder with relative time in attention."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .phasor_hierarchical import Block
from .phasor_hierarchical_v2 import HierarchicalV2Config, PhasorHierarchicalKANV2
from osc_tools.ml.layers.transformer_blocks import FastKANLayer


@dataclass(frozen=True)
class HierarchicalV3Config(HierarchicalV2Config):
    architecture_id: str = "phasor_hierarchical_v3"
    global_span_periods: float = 1500.0
    qk_norm: bool = True
    attention_gate: bool = True


class RotaryBlock(Block):
    """Pre-RMSNorm and full RoPE on the projected queries and keys."""

    def __init__(self, width, heads, cfg, harmonics=0, *, global_time=False):
        super().__init__(width, heads, cfg.grids, cfg.dropout, harmonics)
        head_width = width // heads
        if head_width % 2:
            raise ValueError("RoPE needs an even head width")
        self.n1 = nn.RMSNorm(width, eps=1e-6)
        self.n2 = nn.RMSNorm(width, eps=1e-6)
        self.q_norm = nn.RMSNorm(head_width, eps=1e-6) if cfg.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_width, eps=1e-6) if cfg.qk_norm else nn.Identity()
        self.attention_gate = nn.Linear(width, width) if cfg.attention_gate else None
        if self.attention_gate is not None:
            nn.init.zeros_(self.attention_gate.weight)
            nn.init.zeros_(self.attention_gate.bias)
        shortest = math.pi / (cfg.block_stride / 8) if global_time else 8.
        longest = math.pi / (cfg.global_span_periods if global_time else cfg.block_tokens / 8)
        frequencies = torch.linspace(math.log(shortest), math.log(longest), head_width // 2).exp()
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x, valid, phi, times):
        b, t, d = x.shape
        if self.physics is not None:
            x = x + self.scale[2] * self.physics(phi)
        h = self.n1(x)
        q, k, v = self.qkv(h).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        angles = times.float()[:, None, :, None] * self.frequencies
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

        def rotate(x):
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even*cos-odd*sin, even*sin+odd*cos), dim=-1).flatten(-2)

        attn = F.scaled_dot_product_attention(
            rotate(self.q_norm(q)), rotate(self.k_norm(k)), v,
            attn_mask=valid[:,None,None,:], dropout_p=self.dropout if self.training else 0.,
        ).transpose(1,2).reshape(b,t,d)
        if self.attention_gate is not None:
            attn = attn * (2 * torch.sigmoid(self.attention_gate(h)))
        x = x + self.scale[0] * self.out(attn)
        x = x + self.scale[1] * F.dropout(self.ffn(self.n2(x)), self.dropout, self.training)
        return x * valid.unsqueeze(-1)


class PhasorHierarchicalKANV3(PhasorHierarchicalKANV2):
    def __init__(self, cfg=HierarchicalV3Config()):
        nn.Module.__init__(self)
        self.cfg = cfg
        if cfg.global_span_periods <= 0:
            raise ValueError("Global RoPE span must be positive")
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
        self.channel_blocks = nn.ModuleList(RotaryBlock(c,cfg.channel_heads,cfg) for _ in range(cfg.channel_layers))
        self.mixer = nn.Linear(6*c,d)
        self.raw_bypass = nn.Linear(h*6*2,d,bias=False)
        make = lambda n, **kwargs: nn.ModuleList(RotaryBlock(d,cfg.heads,cfg,h,**kwargs) for _ in range(n))
        self.local_blocks, self.global_blocks, self.fusion_blocks = make(cfg.local_layers),make(cfg.global_layers, global_time=True),make(cfg.fusion_layers)
        self.summary_type = nn.Parameter(torch.randn(2,d)*.02)
        self.context_readout = nn.Linear(2*d,d)
        self.context = nn.ModuleList(nn.Linear(d,d) for _ in self.fusion_blocks)
        self.delta = nn.Linear(d,d)
        self.gate = nn.Linear(2*d,1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,math.log(.05/.95))
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.rates = {}
        for stage in (self.channel_blocks,self.local_blocks,self.global_blocks,self.fusion_blocks):
            for i, block in enumerate(stage): self.rates[id(block)] = cfg.drop_path_rate*(i+1)/len(stage)

    def run_block(self, block, x, valid, times, phi=None):
        rate = self.rates[id(block)] if self.training else 0.
        if rate and torch.rand(()) < rate: return x*valid.unsqueeze(-1)
        y = (checkpoint(block,x,valid,phi,times,use_reentrant=False)
             if self.training and self.cfg.checkpoint_blocks else block(x,valid,phi,times))
        return (x+(y-x)/(1-rate))*valid.unsqueeze(-1) if rate else y

    def forward(self, z, state, times, valid, use_global=True):
        b,t,h,_,_ = z.shape; k,s = self.cfg.block_tokens,self.cfg.block_stride
        n = max(1,math.ceil((t-k)/s)+1); total=(n-1)*s+k; pad=total-t
        z=F.pad(z,(0,0,0,0,0,0,0,pad));state=F.pad(state,(0,0,0,0,0,pad))
        times=F.pad(times,(0,pad));valid=F.pad(valid,(0,pad))
        channels,safe,visible,phi = (checkpoint(self.encode,z,state,use_reentrant=False)
            if self.training and self.cfg.checkpoint_blocks else self.encode(z,state))
        available = (visible | (state == 3)).any(-2)
        valid = valid & available.any(-1)
        index = torch.arange(n,device=z.device)[:,None]*s+torch.arange(k,device=z.device)[None,:]
        mask=valid[:,index]; channel_mask=mask.unsqueeze(-1)&available[:,index]
        x=channels[:,index].permute(0,1,3,2,4).reshape(b*n*6,k,-1)
        cmask=channel_mask.permute(0,1,3,2).reshape(b*n*6,k)
        window_times=times[:,index]; local_times=window_times-window_times[:,:,:1]
        local_times = local_times.reshape(b*n,k)
        channel_times = local_times.repeat_interleave(6, dim=0)
        for block in self.channel_blocks: x=self.run_block(block,x,cmask,channel_times)
        x=x.reshape(b,n,6,k,-1).permute(0,1,3,2,4).flatten(-2).reshape(b*n,k,-1)
        with torch.autocast(z.device.type,enabled=False): x=self.mixer(x.float())
        x=x+self.raw_bypass(safe[:,index].reshape(b*n,k,-1))
        wm=mask.reshape(b*n,k); wp=phi[:,index].reshape(b*n,k,h,3,10)
        for block in self.local_blocks: x=self.run_block(block,x,wm,local_times,wp)
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
        summary=summary+self.summary_type.repeat(n,1)[None]
        mean_phi=(wp*wm[...,None,None,None]).sum(1)/denom[...,None,None]
        # A coordinatewise latent maximum is not a physical phasor: do not invent its physics.
        sphi=torch.stack((mean_phi,torch.zeros_like(mean_phi)),1).reshape(b,2*n,h,3,10)
        for block in self.global_blocks: summary=self.run_block(block,summary,smask,summary_times,sphi)
        context=self.context_readout(summary.reshape(b,n,-1)).reshape(b*n,1,-1).expand(-1,k,-1)
        candidate=local
        for block,project in zip(self.fusion_blocks,self.context):
            candidate=self.run_block(block,candidate+.05*project(context),wm,local_times,wp)
        gate=torch.sigmoid(self.gate(torch.cat((local,context),-1)))
        fused=local+gate*torch.tanh(self.delta(candidate-local))
        sequence=self.norm(self.merge(fused.reshape(b,n,k,-1),mask,index,t))
        return dict(token_features=sequence,local_features=local_sequence,global_features=summary,
                    valid_mask=valid[:,:t],global_valid_mask=smask,times=times[:,:t],global_times=summary_times,
                    context_gate=self.merge(gate.reshape(b,n,k,1),mask,index,t))
