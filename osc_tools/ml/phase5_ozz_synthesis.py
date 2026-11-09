"""On-the-fly RTDS-only contexts. All edits stay within the first/last source period."""
import math
import numpy as np


class UnsuitableEdges(ValueError):
    pass


def periodic_template(raw,spp):
    grid=np.unique(np.r_[0.,np.arange(1,int(np.floor(spp))+1),spp])
    values=np.stack([np.interp(grid,np.arange(len(raw)),raw[:,c]) for c in range(8)],1)
    mismatch=values[-1]-values[0]
    values=values-(grid/spp)[:,None]*mismatch
    return grid,values


def compose_context(raw, labels, spp, total_samples, rng, label_tail_periods=1, include_events=True):
    if len(raw)<2*spp or total_samples<len(raw): raise UnsuitableEdges('Source/context too short')
    edge=int(math.ceil(spp))
    # Exactly ONE original source period at either boundary, not two.
    if np.asarray(labels[:edge]).any() or (include_events and np.asarray(labels[-edge:]).any()):
        raise UnsuitableEdges('Positive labels inside the first/last source period')
    if not np.isfinite(raw[:,[0,1,2,4,5,6]]).all(): raise UnsuitableEdges('Incomplete source phases')
    seconds=total_samples/(spp*50.)
    reserve=max(len(raw),int(math.ceil(.5*spp*50)))
    count=min(max(1,int(math.ceil(seconds/3))),total_samples//reserve)
    gaps=rng.multinomial(total_samples-count*reserve,rng.dirichlet(np.full(count+1,.65)))
    starts=np.cumsum(gaps[:-1])+np.arange(count)*reserve
    origin=float(rng.uniform(0,spp));grid,template=periodic_template(raw,spp)
    phase=np.remainder(np.arange(total_samples)+origin,spp)
    background=np.stack([np.interp(phase,grid,template[:,c]) for c in range(8)],1).astype(np.float32)
    wave=background.copy();target=np.zeros((total_samples,4),np.bool_);placements=[]
    if not include_events: return wave,target,[]
    source_end=len(raw)-1
    runs=[]
    for c in range(4):
        changes=np.diff(np.r_[False,np.asarray(labels[:,c])>.5,False].astype(np.int8))
        runs.append(list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))))
    for reserved in starts:
        shifts=np.arange(int(math.ceil(spp))+1)
        offsets=np.remainder(reserved+shifts+origin,spp)
        duration=np.floor((source_end-offsets)/spp)*spp
        right_span=offsets+duration-(source_end-spp)
        feasible=(spp-offsets>=spp/8)&(right_span>=spp/8)&(shifts+duration<=reserve)
        choices=shifts[feasible]
        if not len(choices): raise UnsuitableEdges('No phase-aligned crop within the source boundary periods')
        shift=int(rng.choice(choices));start=int(reserved+shift);a=float(offsets[shift]);length=float(duration[shift])
        n=int(math.floor(length));q=a+np.arange(n);end=start+n
        copy=np.stack([np.interp(q,np.arange(len(raw)),raw[:,c]) for c in range(8)],1)
        # The total crop+blend is bounded by ONE period in ORIGINAL coordinates.
        left_progress=np.clip((q-a)/(spp-a),0,1)
        right_progress=np.clip((a+length-q)/(a+length-(source_end-spp)),0,1)
        alpha=.5-.5*np.cos(np.pi*np.minimum(left_progress,right_progress))
        wave[start:end]=background[start:end]+alpha[:,None]*(copy-background[start:end])
        for c,intervals in enumerate(runs):
            for lo,hi in intervals:
                l=max(0,start+int(math.floor(lo-a)))
                r=min(total_samples,start+int(math.ceil(hi-a+label_tail_periods*spp)))
                if l<r: target[l:r,c]=True
        placements.append(dict(reserve_start=int(reserved),reserve_end=int(reserved+reserve),
            start=start,end=end,source_crop_start=a,source_crop_end=a+length,
            left_blend_source_end=float(spp),right_blend_source_start=float(source_end-spp)))
    assert all(x['reserve_end']<=y['reserve_start'] for x,y in zip(placements,placements[1:]))
    return wave,target,placements
