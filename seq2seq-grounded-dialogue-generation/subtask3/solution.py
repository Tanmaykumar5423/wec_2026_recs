"""Dual-source recurrent decoder for grounded Hinglish dialogue.

This file provides the core neural architecture. Dataset-specific field extraction and
batch construction should follow the CMU Hinglish DoG schema used in the notebook.
"""
from __future__ import annotations
import torch
from torch import nn

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=256):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.gru=nn.GRU(emb_dim,hidden,batch_first=True)
    def forward(self,x):
        return self.gru(self.embedding(x))

class AdditiveAttention(nn.Module):
    def __init__(self,enc_dim,query_dim,attn_dim=128):
        super().__init__()
        self.enc=nn.Linear(enc_dim,attn_dim,bias=False)
        self.query=nn.Linear(query_dim,attn_dim,bias=False)
        self.v=nn.Linear(attn_dim,1,bias=False)
    def forward(self,states,query,mask=None):
        score=self.v(torch.tanh(self.enc(states)+self.query(query).unsqueeze(1))).squeeze(-1)
        if mask is not None:
            score=score.masked_fill(~mask,-1e9)
        weights=torch.softmax(score,-1)
        context=torch.bmm(weights.unsqueeze(1),states).squeeze(1)
        return context,weights

class DualSourceDecoder(nn.Module):
    def __init__(self,vocab_size,emb_dim=256,hidden=256):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.hist_attn=AdditiveAttention(hidden,hidden)
        self.doc_attn=AdditiveAttention(hidden,hidden)
        self.gru=nn.GRU(emb_dim+2*hidden,hidden,batch_first=True)
        self.out=nn.Linear(hidden,vocab_size)
    def forward_step(self,token,history_states,document_states,hidden_state,
                     history_mask=None,document_mask=None):
        query=hidden_state[-1]
        hctx,_=self.hist_attn(history_states,query,history_mask)
        dctx,_=self.doc_attn(document_states,query,document_mask)
        emb=self.embedding(token).unsqueeze(1)
        inp=torch.cat([emb.squeeze(1),hctx,dctx],dim=-1).unsqueeze(1)
        out,new_hidden=self.gru(inp,hidden_state)
        return self.out(out.squeeze(1)),new_hidden

def grounding_overlap(reference,generated,document):
    ref=set(reference.lower().split())
    gen=set(generated.lower().split())
    doc=set(document.lower().split())
    relevant=ref & doc
    return len(gen & relevant)/max(1,len(relevant))
