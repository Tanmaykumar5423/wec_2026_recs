"""Attention module and decoding utilities for Subtask 2."""
from __future__ import annotations
import torch
from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, hidden=128):
        super().__init__()
        self.w_enc=nn.Linear(enc_dim,hidden,bias=False)
        self.w_dec=nn.Linear(dec_dim,hidden,bias=False)
        self.v=nn.Linear(hidden,1,bias=False)
    def forward(self, enc_states, dec_state, mask=None):
        score=self.v(torch.tanh(self.w_enc(enc_states)+self.w_dec(dec_state).unsqueeze(1))).squeeze(-1)
        if mask is not None: score=score.masked_fill(~mask,-1e9)
        weights=torch.softmax(score,-1)
        context=torch.bmm(weights.unsqueeze(1),enc_states).squeeze(1)
        return context,weights

def length_normalized(logprob,length,alpha=.7):
    return logprob/(((5+length)/6)**alpha)

def greedy(step_fn,bos_id,eos_id,max_len=64):
    ids=[bos_id]
    state=None
    for _ in range(max_len):
        nxt,state=step_fn(ids[-1],state)
        ids.append(int(nxt))
        if ids[-1]==eos_id: break
    return ids

def beam_search(step_fn,bos_id,eos_id,beam_size=5,max_len=64,alpha=.7):
    beams=[([bos_id],0.0,None)]
    for _ in range(max_len):
        candidates=[]
        finished=True
        for ids,score,state in beams:
            if ids[-1]==eos_id:
                candidates.append((ids,score,state)); continue
            finished=False
            log_probs,new_state=step_fn(ids[-1],state)
            values,indices=torch.topk(log_probs,beam_size)
            for lp,idx in zip(values.tolist(),indices.tolist()):
                candidates.append((ids+[int(idx)],score+float(lp),new_state))
        if finished: break
        candidates.sort(key=lambda x:length_normalized(x[1],len(x[0]),alpha),reverse=True)
        beams=candidates[:beam_size]
    return max(beams,key=lambda x:length_normalized(x[1],len(x[0]),alpha))[0]
