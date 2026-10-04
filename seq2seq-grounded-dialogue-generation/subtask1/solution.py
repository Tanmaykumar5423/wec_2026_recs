"""Task 1 training-ready seq2seq model for TSV parallel data.
Source and target files contain one sentence per line.
"""
from __future__ import annotations
import argparse, random, math
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import corpus_bleu

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

class Vocab:
    def __init__(self, lines, max_size=20000):
        counts = Counter(tok for line in lines for tok in line.lower().split())
        words = [w for w,_ in counts.most_common(max_size-len(SPECIAL))]
        self.itos = SPECIAL + words
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, text, limit=64):
        toks = text.lower().split()[:limit-2]
        return [1] + [self.stoi.get(t,3) for t in toks] + [2]

class PairDataset(Dataset):
    def __init__(self, src, tgt, sv, tv, max_len=64):
        self.x=[sv.encode(s,max_len) for s in src]
        self.y=[tv.encode(t,max_len) for t in tgt]
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.y[i]

def collate(batch):
    xs,ys=zip(*batch)
    xm=max(map(len,xs)); ym=max(map(len,ys))
    X=torch.zeros(len(xs),xm,dtype=torch.long); Y=torch.zeros(len(ys),ym,dtype=torch.long)
    for i,s in enumerate(xs): X[i,:len(s)]=torch.tensor(s)
    for i,t in enumerate(ys): Y[i,:len(t)]=torch.tensor(t)
    return X,Y

class Encoder(nn.Module):
    def __init__(self,vocab,emb,hid):
        super().__init__(); self.e=nn.Embedding(vocab,emb,padding_idx=0); self.r=nn.GRU(emb,hid,batch_first=True)
    def forward(self,x): return self.r(self.e(x))

class Decoder(nn.Module):
    def __init__(self,vocab,emb,hid):
        super().__init__(); self.e=nn.Embedding(vocab,emb,padding_idx=0); self.r=nn.GRU(emb,hid,batch_first=True); self.o=nn.Linear(hid,vocab)
    def forward(self,y,h):
        z,h=self.r(self.e(y),h); return self.o(z),h

def train_epoch(enc,dec,loader,opt,criterion,device,tf=.5):
    enc.train(); dec.train(); total=0
    for X,Y in loader:
        X,Y=X.to(device),Y.to(device); opt.zero_grad()
        _,h=enc(X); inp=Y[:,0]; loss=0
        for t in range(1,Y.size(1)):
            out,h=dec(inp.unsqueeze(1),h); loss=loss+criterion(out.squeeze(1),Y[:,t])
            inp=Y[:,t] if random.random()<tf else out.argmax(-1).squeeze(1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),1.0); opt.step()
        total += float(loss)
    return total/max(1,len(loader))

@torch.no_grad()
def generate(enc,dec,src,vocab_s,vocab_t,device,max_len=64):
    x=torch.tensor([vocab_s.encode(src)]).to(device); _,h=enc(x); tok=1; out=[]
    for _ in range(max_len):
        logits,h=dec(torch.tensor([[tok]],device=device),h); tok=int(logits.argmax(-1))
        if tok==2: break
        out.append(vocab_t.itos[tok])
    return " ".join(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",required=True); ap.add_argument("--tgt",required=True); ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--batch-size",type=int,default=64); args=ap.parse_args()
    random.seed(42); torch.manual_seed(42); device="cuda" if torch.cuda.is_available() else "cpu"
    src=open(args.src,encoding="utf8").read().splitlines(); tgt=open(args.tgt,encoding="utf8").read().splitlines()
    n=min(len(src),len(tgt)); src,tgt=src[:n],tgt[:n]; cut=int(.9*n); train_s,train_t=src[:cut],tgt[:cut]; test_s,test_t=src[cut:],tgt[cut:]
    sv,tv=Vocab(train_s),Vocab(train_t); ds=PairDataset(train_s,train_t,sv,tv); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    enc,dec=Encoder(len(sv.itos),256,256).to(device),Decoder(len(tv.itos),256,256).to(device); opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),1e-3); crit=nn.CrossEntropyLoss(ignore_index=0)
    for e in range(args.epochs): print("epoch",e+1,"loss",train_epoch(enc,dec,loader,opt,crit,device))
    for s in test_s[:10]: print(s,"=>",generate(enc,dec,s,sv,tv,device))

if __name__=="__main__": main()
