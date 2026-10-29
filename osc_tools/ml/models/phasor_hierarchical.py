"""Phase 5 PhasorHierarchicalKAN v1. Legacy transformer is unchanged.

Physical quantities never pass through latent normalization. Arithmetic edges
are exclusively U_s,h--I_s,h. Cross-frequency mixing starts after the pair stem
and is latent only. Local features never read another ten-period block.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from osc_tools.ml.layers.transformer_blocks import FastKANLayer


@dataclass(frozen=True)
class HierarchicalConfig:
    architecture_id: str = "phasor_hierarchical_kan_v1"
    harmonics: int = 13  # h1..h9, lp2,lp4,lp6,lp10
    channel_width: int = 32
    width: int = 128
    heads: int = 4
    channel_layers: int = 4
    local_layers: int = 12
    global_layers: int = 12
    fusion_layers: int = 4
    block_tokens: int = 80  # ten periods, eight samples per period
    grids: int = 5
    dropout: float = 0.05
    checkpoint_blocks: bool = True


class KANFFN(nn.Module):
    def __init__(self, width, grids):
        super().__init__()
        self.up = FastKANLayer(width, 2*width, num_grids=grids, use_layernorm=False)
        self.down = FastKANLayer(2*width, width, num_grids=grids, use_layernorm=False)

    def forward(self, x):
        return self.down(self.up(x))


def physical_features(z, visible):
    """z [..., H, 6, 2], output [..., H, 3, 10], FP32.

    S uses peak-phasor convention: U*conj(I)/2. It is a per-component
    power proxy, not total three-phase power. Z/Y are regularized p.u. ratios.
    """
    z = z.float()
    i, u = z[..., :3, :], z[..., 3:, :]
    valid = visible[..., :3] & visible[..., 3:]
    p = (u*i).sum(-1) * .5
    q = (u[..., 1]*i[..., 0] - u[..., 0]*i[..., 1]) * .5
    mag = torch.sqrt(p*p + q*q + 1e-16)
    i2, u2 = i.square().sum(-1), u.square().sum(-1)
    # Existing source scales include current reserve=20 and voltage reserve=3.
    # Explicit numerical thresholds in these prepared units, versioned here.
    zi_valid, yi_valid = valid & (i2 > (1/2000)**2), valid & (u2 > (1/300)**2)
    zr, zq = 2*p/(i2+1e-6), 2*q/(i2+1e-6)
    yr, yq = 2*p/(u2+1e-5), -2*q/(u2+1e-5)
    compress = lambda x: torch.sign(x)*torch.log1p(x.abs())
    result = torch.stack((compress(p), compress(q), torch.log1p(mag),
                          p/mag, q/mag,
                          compress(zr)*zi_valid, compress(zq)*zi_valid,
                          compress(yr)*yi_valid, compress(yq)*yi_valid,
                          valid.float()), -1)
    return result * valid.unsqueeze(-1)


class PhysicalInjection(nn.Module):
    """Repeated vector/directional chain, with a learned MTA per component."""
    def __init__(self, harmonics, width):
        super().__init__()
        self.mta = nn.Parameter(torch.zeros(harmonics, 3))
        self.vector = nn.Linear(harmonics*3*10, width, bias=False)
        self.direction = nn.Linear(harmonics*3, width, bias=False)

    def forward(self, phi):
        directional = (phi[..., 3]*self.mta.cos() + phi[..., 4]*self.mta.sin())
        directional = torch.tanh(3*directional)*phi[..., 2]*phi[..., 9]
        return self.vector(phi.flatten(-3)) + self.direction(directional.flatten(-2))


class Block(nn.Module):
    def __init__(self, width, heads, grids, dropout, harmonics=0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ffn = KANFFN(width, grids)
        self.scale = nn.Parameter(torch.full((3, width), .05))
        self.physics = PhysicalInjection(harmonics, width) if harmonics else None

    def forward(self, x, valid, phi=None):
        b,t,d = x.shape
        if self.physics is not None:
            x = x + self.scale[2]*self.physics(phi)
        q,k,v = self.qkv(self.n1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        attn = F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],
                                             dropout_p=self.dropout if self.training else 0.)
        x = x + self.scale[0]*self.out(attn.transpose(1,2).reshape(b,t,d))
        x = x + self.scale[1]*F.dropout(self.ffn(self.n2(x)),self.dropout,self.training)
        return x*valid.unsqueeze(-1)


def positional(time, width):
    freq = torch.exp(torch.arange(0,width,2,device=time.device).float()*(-math.log(10000)/width))
    a = time.float().unsqueeze(-1)*freq
    return torch.stack((a.sin(),a.cos()),-1).flatten(-2)


class PhasorHierarchicalKAN(nn.Module):
    def __init__(self, cfg=HierarchicalConfig()):
        super().__init__()
        self.cfg = cfg
        h,cw,d = cfg.harmonics,cfg.channel_width,cfg.width
        if cw % cfg.heads or d % cfg.heads or cw % 2 or d % 2:
            raise ValueError("Widths must be even and divisible by heads")
        # Each pair is encoded independently: Re/Im, log amplitude, four states.
        self.pair = nn.Sequential(FastKANLayer(7,16,num_grids=cfg.grids,use_layernorm=False),
                                  FastKANLayer(16,16,num_grids=cfg.grids,use_layernorm=False))
        self.harmonic_mix = nn.Linear(h*16,cw)
        self.channel_id = nn.Parameter(torch.randn(6,cw)*.02)
        self.channel_blocks = nn.ModuleList(Block(cw,cfg.heads,cfg.grids,cfg.dropout) for _ in range(cfg.channel_layers))
        self.mixer = nn.Linear(6*cw,d)
        self.raw_bypass = nn.Linear(h*6*2,d,bias=False)
        make = lambda n: nn.ModuleList(Block(d,cfg.heads,cfg.grids,cfg.dropout,h) for _ in range(n))
        self.local_blocks, self.global_blocks, self.fusion_blocks = make(cfg.local_layers),make(cfg.global_layers),make(cfg.fusion_layers)
        self.context = nn.ModuleList(nn.Linear(d,d) for _ in range(cfg.fusion_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,h*6*2)
        self.local_head = nn.Linear(d,h*6*2)

    def run_block(self, block, x, valid, phi=None):
        if self.training and self.cfg.checkpoint_blocks:
            return checkpoint(block,x,valid,phi,use_reentrant=False)
        return block(x,valid,phi)

    def encode_channels(self, z, state):
        """No U/I or sequence mixing here; no cross-harmonic products anywhere."""
        visible = (state==1)|(state==2)
        safe = torch.where(visible.unsqueeze(-1),z,torch.zeros_like(z))
        amp = torch.sqrt(safe.square().sum(-1)+1e-16)
        onehot = F.one_hot(state.long(),4).float()
        x = torch.cat((safe,torch.log1p(amp).unsqueeze(-1),onehot),-1)
        x = self.pair(x)  # [..., H,6,16]; two per-component KAN layers
        x = x.transpose(-3,-2).flatten(-2)  # harmonic mixing only now
        return self.harmonic_mix(x)+self.channel_id, safe, visible

    def forward(self, z, state, times, valid, local_only=False):
        """Re/Im input B,T,H,6,2; states missing=0/measured=1/derived=2/hidden=3.

        Offline global context. local_only returns features for independent
        ten-period RNM windows (no future windows, but bidirectional within one).
        """
        b,t,h,_,_ = z.shape
        k = self.cfg.block_tokens
        n = (t+k-1)//k
        pad = n*k-t
        if pad:
            z = F.pad(z,(0,0,0,0,0,0,0,pad)); state = F.pad(state,(0,0,0,0,0,pad))
            times = F.pad(times,(0,pad)); valid = F.pad(valid,(0,pad))
        # Stem is checkpointed too: its per-pair RBF activations can be large.
        if self.training and self.cfg.checkpoint_blocks:
            channels,safe,visible = checkpoint(self.encode_channels,z,state,use_reentrant=False)
        else:
            channels,safe,visible = self.encode_channels(z,state)
        phi = physical_features(safe,visible).reshape(b*n,k,h,3,10)
        mask = valid.reshape(b*n,k)
        local_time = times.reshape(b*n,k)
        local_time = local_time-local_time[:,:1]
        x = channels.reshape(b,n,k,6,-1).permute(0,1,3,2,4).reshape(b*n*6,k,-1)
        cmask = mask[:,None,:].expand(-1,6,-1).reshape(b*n*6,k)
        pe = positional(local_time,self.cfg.channel_width)
        x = x+pe[:,None,:,:].expand(-1,6,-1,-1).reshape_as(x)
        for block in self.channel_blocks:
            x = self.run_block(block,x,cmask)
        x = x.reshape(b,n,6,k,-1).permute(0,1,3,2,4).flatten(-2).reshape(b*n,k,-1)
        x = self.mixer(x)+self.raw_bypass(safe.reshape(b*n,k,-1))
        for block in self.local_blocks:
            x = self.run_block(block,x,mask,phi)
        local = x
        local_pred = self.local_head(self.norm(local)).reshape(b,n*k,h,6,2)[:,:t]
        if local_only:
            return {"local_features":local.reshape(b,n*k,-1)[:,:t],"prediction":local_pred}
        denom = mask.sum(-1,keepdim=True).clamp_min(1)
        summary = (x*mask.unsqueeze(-1)).sum(1)/denom
        summary = summary.reshape(b,n,-1)
        smask = mask.any(-1).reshape(b,n)
        sphi = (phi*mask[...,None,None,None]).sum(1)/denom[...,None,None]
        sphi = sphi.reshape(b,n,h,3,10)
        summary = summary + positional(times.reshape(b,n,k)[:,:,0],self.cfg.width)
        for block in self.global_blocks:
            summary = self.run_block(block,summary,smask,sphi)
        summary = summary.reshape(b*n,1,-1)
        for block,project in zip(self.fusion_blocks,self.context):
            x = x+.05*project(summary)
            x = self.run_block(block,x,mask,phi)
        pred = self.head(self.norm(x)).reshape(b,n*k,h,6,2)[:,:t]
        return {"prediction":pred,"local_prediction":local_pred,
                "local_features":local.reshape(b,n*k,-1)[:,:t]}


def vector_loss(pred, target, selected, beta=.1):
    """Radial Huber of Euclidean vector distance; rotation invariant.

    Returns sum and count for correct accumulation across variable lengths.
    """
    diff = pred.float()-target.float()
    squared = diff.square().sum(-1)
    distance = torch.sqrt(squared.clamp_min(1e-12))
    loss = torch.where(squared < beta*beta, squared/(2*beta), distance-beta/2)
    return (loss*selected).sum(), selected.sum()
