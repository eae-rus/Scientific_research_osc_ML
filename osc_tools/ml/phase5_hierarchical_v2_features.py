"""Six observed phases; U1 -> I1 -> zero angle is only a shared reference."""
import math
import numpy as np
import torch
from osc_tools.ml.phase5_hierarchical_data import ExactFrequencyBuilder, token_positions

CURRENT_NOISE = 1/2000
VOLTAGE_NOISE = 1/300


def phase_window(raw, labels, spp, blocks=None, basis='phase', provenance=None, history=None):
    raw=np.asarray(raw,dtype=np.float32).copy()
    history=np.empty((0,8),np.float32) if history is None else np.asarray(history,dtype=np.float32).copy()
    provenance=(np.isfinite(raw).any(0).astype(np.uint8) if provenance is None else np.asarray(provenance).copy())
    # No inference of a missing phase from other phases/line voltages.
    if basis != 'phase': provenance[4:7]=0
    provenance[provenance==2]=0
    raw[:,provenance==0]=np.nan; history[:,provenance==0]=np.nan
    joined=np.concatenate((history,raw)); count=int(math.ceil(len(raw)*8/spp))
    positions=token_positions(0,count,spp);positions=positions[positions<len(raw)]
    if not len(positions) or positions[-1]!=len(raw)-1:
        positions=np.r_[positions,len(raw)-1]
    absolute=positions+len(history); rounded=int(math.floor(spp+.5))
    specs=((1,9),(2,1),(4,1),(6,1),(10,1)); builder=ExactFrequencyBuilder(spp)
    values=np.full((len(positions),13,8),np.nan+1j*np.nan,np.complex64)
    for first in range(0,len(raw),65536):
        chosen=np.flatnonzero((positions>=first)&(positions<first+65536))
        if not len(chosen): continue
        ends=absolute[chosen];left=max(0,int(ends[0])-int(math.floor(10*spp+.5))+1);right=int(ends[-1])+1
        destination=0
        for periods,harmonics in specs:
            p=builder._phasors_prefix(joined[left:right],ends-left,rounded*periods,harmonics)
            values[chosen,destination:destination+harmonics]=p.transpose(0,2,1)
            destination+=harmonics
    phases=values[:,:,[0,1,2,4,5,6]]
    available=np.isfinite(phases.real)&np.isfinite(phases.imag)
    available &= (provenance[[0,1,2,4,5,6]]!=0)[None,None,:]
    # A zero is used ONLY in this reference calculation, never as an invented input.
    h1=np.where(available[:,0],phases[:,0],0.)
    a=np.exp(2j*np.pi/3); coefficients=np.array([1,a,a*a])/3
    i1=h1[:,:3]@coefficients;u1=h1[:,3:]@coefficients
    use_u=np.abs(u1)>VOLTAGE_NOISE;use_i=(~use_u)&(np.abs(i1)>CURRENT_NOISE)
    reference=np.where(use_u,u1,np.where(use_i,i1,1.+0j))
    rotation=np.conj(reference/np.maximum(np.abs(reference),1e-30))
    phases=np.where(available,phases,0.)*rotation[:,None,None]
    z=np.stack((phases.real,phases.imag),-1).astype(np.float32)
    state=available.astype(np.int64)
    starts=np.r_[0,positions[:-1]+1]
    target=np.maximum.reduceat(labels,starts,axis=0).astype(np.float32)
    valid=available.any((1,2))
    return dict(z=torch.from_numpy(z),state=torch.from_numpy(state),target=torch.from_numpy(target),
                times=torch.tensor((positions+1)/spp,dtype=torch.float32),
                valid=torch.from_numpy(valid),supervised=torch.from_numpy(valid.copy()),
                sample_ends=torch.tensor(positions+1,dtype=torch.int64),
                reference_source=torch.tensor(np.where(use_u,1,np.where(use_i,2,0)),dtype=torch.int64))
