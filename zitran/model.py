"""ZITRAN Transformer — from scratch in pure NumPy.
Multi-head self-attention, positional embeddings, pre-norm blocks, GELU FFN,
manual backprop, Adam, gradient clipping, cosine LR. No autograd, no frameworks.
Architecture: CLS pooling -> multi-task heads (threat class + binary threat).
"""
import numpy as np

def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def gelu_grad(x):
    t = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
    sech2 = 1 - t**2
    return 0.5*(1+t) + 0.5*x*sech2*np.sqrt(2/np.pi)*(1 + 3*0.044715*x**2)
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

class LayerNorm:
    def __init__(self, d, rng): self.g=np.ones(d); self.b=np.zeros(d); self.eps=1e-5
    def forward(self, x):
        self.mu=x.mean(-1,keepdims=True); self.var=x.var(-1,keepdims=True)
        self.xhat=(x-self.mu)/np.sqrt(self.var+self.eps); self.std=np.sqrt(self.var+self.eps)
        return self.g*self.xhat+self.b
    def backward(self, dy):
        N=dy.shape[-1]; dxhat=dy*self.g
        dvar=np.sum(dxhat*self.xhat*-0.5/self.std,axis=-1,keepdims=True)
        dmu=np.sum(dxhat*-1/self.std,axis=-1,keepdims=True)+dvar*np.mean(-2*self.xhat,axis=-1,keepdims=True)
        self.dg=np.sum(dy*self.xhat,axis=(0,1)); self.db=np.sum(dy,axis=(0,1))
        return dxhat/self.std + dvar*2*self.xhat/N + dmu/N

class Block:
    def __init__(self, d, heads, ff, rng):
        s=lambda *sh: rng.normal(0, np.sqrt(2.0/(sh[0]+sh[1] if len(sh)>1 else sh[0])), sh).astype(np.float64)
        self.d, self.h, self.dh = d, heads, d//heads
        self.Wq,self.Wk,self.Wv,self.Wo = s(d,d),s(d,d),s(d,d),s(d,d)
        self.bq,self.bk,self.bv,self.bo = np.zeros(d),np.zeros(d),np.zeros(d),np.zeros(d)
        self.W1,self.b1,self.W2,self.b2 = s(d,ff),np.zeros(ff),s(ff,d),np.zeros(d)
        self.ln1,self.ln2 = LayerNorm(d,rng),LayerNorm(d,rng)
    def params(self):
        p={'Wq':self.Wq,'Wk':self.Wk,'Wv':self.Wv,'Wo':self.Wo,'bq':self.bq,'bk':self.bk,'bv':self.bv,'bo':self.bo,
           'W1':self.W1,'b1':self.b1,'W2':self.W2,'b2':self.b2,'ln1g':self.ln1.g,'ln1b':self.ln1.b,'ln2g':self.ln2.g,'ln2b':self.ln2.b}
        return p
    def load_params(self,p):
        for k,v in p.items():
            if k=='ln1g': self.ln1.g=v
            elif k=='ln1b': self.ln1.b=v
            elif k=='ln2g': self.ln2.g=v
            elif k=='ln2b': self.ln2.b=v
            else: setattr(self,k,v)
    def forward(self,x, mask):
        B,T,D = x.shape; H, dh = self.h, self.dh
        self.x_in=x
        h=self.ln1.forward(x)
        Q=h@self.Wq+self.bq; K=h@self.Wk+self.bk; V=h@self.Wv+self.bv
        def split(z): return z.reshape(B,T,H,dh).transpose(0,2,1,3)
        Qh,Kh,Vh = split(Q),split(K),split(V)
        scores=Qh@Kh.transpose(0,1,3,2)/np.sqrt(dh)
        # mask padded positions (mask: B,T 1=keep)
        m = mask[:,None,None,:]
        scores = np.where(m>0, scores, -1e9)
        attn=softmax(scores,axis=-1)
        ctx=attn@Vh  # B,H,T,dh
        ctx_r=ctx.transpose(0,2,1,3).reshape(B,T,D)
        attn_out=ctx_r@self.Wo+self.bo
        x2=x+attn_out
        h2=self.ln2.forward(x2)
        hidden=gelu(h2@self.W1+self.b1)
        ff_out=hidden@self.W2+self.b2
        out=x2+ff_out
        self.cache=dict(h=h,Qh=Qh,Kh=Kh,Vh=Vh,attn=attn,ctx_r=ctx_r,h2=h2,hidden=hidden,x2=x2)
        return out
    def backward(self, dout, mask):
        c=self.cache; B,T,D=dout.shape; H,dh=self.h,self.dh
        g={}
        # FFN branch
        dff=dout; dx2=dout.copy()
        g['W2']=c['h2'].reshape(-1,D).T @ (dff.reshape(-1,D)) if False else (c['hidden'].reshape(-1,c['hidden'].shape[-1]).T @ dff.reshape(-1,D))
        g['b2']=dff.sum(axis=(0,1))
        dhidden=dff@self.W2.T
        dpre=dhidden*gelu_grad(c['h2']@self.W1+self.b1)
        g['W1']=c['h2'].reshape(-1,D).T @ dpre.reshape(-1,dpre.shape[-1]); g['b1']=dpre.sum(axis=(0,1))
        dh2=dpre@self.W1.T
        dx2 += self.ln2.backward(dh2); g['ln2g']=self.ln2.dg; g['ln2b']=self.ln2.db
        # attention branch
        dattn_out=dx2; dx=dx2.copy()
        g['Wo']=c['ctx_r'].reshape(-1,D).T @ dattn_out.reshape(-1,D); g['bo']=dattn_out.sum(axis=(0,1))
        dctx_r=dattn_out@self.Wo.T
        dctx=dctx_r.reshape(B,T,H,dh).transpose(0,2,1,3)
        attn=c['attn']
        dVh=attn.transpose(0,1,3,2)@dctx
        dattn=dctx@c['Vh'].transpose(0,1,3,2)
        dscores=attn*(dattn-np.sum(dattn*attn,axis=-1,keepdims=True))/np.sqrt(dh)
        dQh=dscores@c['Kh']; dKh=dscores.transpose(0,1,3,2)@c['Qh']
        def merge(z): return z.transpose(0,2,1,3).reshape(B,T,D)
        dQ,dK,dV = merge(dQh),merge(dKh),merge(dVh)
        h=c['h']
        g['Wq']=h.reshape(-1,D).T@dQ.reshape(-1,D); g['bq']=dQ.sum(axis=(0,1))
        g['Wk']=h.reshape(-1,D).T@dK.reshape(-1,D); g['bk']=dK.sum(axis=(0,1))
        g['Wv']=h.reshape(-1,D).T@dV.reshape(-1,D); g['bv']=dV.sum(axis=(0,1))
        dh=dQ@self.Wq.T+dK@self.Wk.T+dV@self.Wv.T
        dx += self.ln1.backward(dh); g['ln1g']=self.ln1.dg; g['ln1b']=self.ln1.db
        self.grads=g
        return dx

class ZITRAN:
    def __init__(self, vocab, n_classes=10, max_len=40, d=128, heads=4, ff=256, layers=3, seed=7):
        rng=np.random.default_rng(seed)
        self.vocab,self.d,self.max_len,self.n_classes=vocab,d,max_len,n_classes
        self.emb=rng.normal(0,0.05,(vocab,d)); self.pos=rng.normal(0,0.02,(max_len,d))
        self.blocks=[Block(d,heads,ff,rng) for _ in range(layers)]
        self.lnf=LayerNorm(d,rng)
        self.Wc=rng.normal(0,np.sqrt(2/(d+n_classes)),(d,n_classes)); self.bc=np.zeros(n_classes)
        self.Wb=rng.normal(0,np.sqrt(2/(d+2)),(d,2)); self.bb=np.zeros(2)
        self._init_adam()
    # flat param registry
    def named_params(self):
        p={'emb':self.emb,'pos':self.pos,'lnfg':self.lnf.g,'lnfb':self.lnf.b,'Wc':self.Wc,'bc':self.bc,'Wb':self.Wb,'bb':self.bb}
        for i,b in enumerate(self.blocks):
            for k,v in b.params().items(): p[f'block{i}.{k}']=v
        return p
    def set_param(self,name,v):
        if '.' in name and name.startswith('block'):
            i,k=name.split('.',1); self.blocks[int(i[5:])].load_params({k:v}) if False else None
            blk=self.blocks[int(i[5:])]
            if k=='ln1g': blk.ln1.g=v
            elif k=='ln1b': blk.ln1.b=v
            elif k=='ln2g': blk.ln2.g=v
            elif k=='ln2b': blk.ln2.b=v
            else: setattr(blk,k,v)
        elif name=='emb': self.emb=v
        elif name=='pos': self.pos=v
        elif name=='lnfg': self.lnf.g=v
        elif name=='lnfb': self.lnf.b=v
        else: setattr(self,name,v)
    def _init_adam(self):
        self.adam_m={k:np.zeros_like(v) for k,v in self.named_params().items()}
        self.adam_v={k:np.zeros_like(v) for k,v in self.named_params().items()}
        self.step=0
    def forward(self, x):
        B,T=x.shape
        self.mask=(x!=0).astype(np.float64)
        h=self.emb[x]+self.pos[:T][None,:,:]
        self.x_ids=x
        for b in self.blocks: h=b.forward(h,self.mask)
        h=self.lnf.forward(h)
        cls=h[:,0,:]
        logits=cls@self.Wc+self.bc
        bin_logits=cls@self.Wb+self.bb
        self.cls=cls
        return logits, bin_logits
    def loss_and_grads(self, x, y, ybin, lam=0.5, smooth=0.05, class_w=None):
        logits, bl = self.forward(x); B=x.shape[0]
        probs=softmax(logits); bprobs=softmax(bl)
        eps=1e-12
        loss = -np.log(probs[np.arange(B),y]+eps).mean() + lam*-np.log(bprobs[np.arange(B),ybin]+eps).mean()
        w = class_w[y].astype(np.float64) if class_w is not None else np.ones(B)
        w = w / w.mean()
        dlogits=probs.copy()
        target=np.full_like(probs, smooth/probs.shape[1]); target[np.arange(B),y]+=1-smooth
        dlogits=(dlogits-target)*(w/B)[:,None]
        dbl=bprobs.copy(); dbl[np.arange(B),ybin]-=1; dbl*= lam/B
        grads={}
        grads['Wc']=self.cls.T@dlogits; grads['bc']=dlogits.sum(0)
        grads['Wb']=self.cls.T@dbl; grads['bb']=dbl.sum(0)
        dcls=dlogits@self.Wc.T+dbl@self.Wb.T
        dh=np.zeros((B,x.shape[1],self.d)); dh[:,0,:]=dcls
        dh=self.lnf.backward(dh); grads['lnfg']=self.lnf.dg; grads['lnfb']=self.lnf.db
        for b in reversed(self.blocks):
            dh=b.backward(dh,self.mask)
            for k,v in b.grads.items(): grads[f'block{self.blocks.index(b)}.{k}']=v
        demb=np.zeros_like(self.emb); np.add.at(demb, self.x_ids, dh)
        grads['emb']=demb; grads['pos']=dh.sum(axis=0)
        self._last_grads=grads
        acc=(logits.argmax(1)==y).mean()
        return float(loss), float(acc), grads
    def adam_update(self, lr=3e-3, clip=1.0):
        grads=self._last_grads
        # clip global norm
        norm=np.sqrt(sum(float((g*g).sum()) for g in grads.values()))
        scale=min(1.0, clip/(norm+1e-12))
        self.step+=1; b1,b2,ep=0.9,0.999,1e-8
        for name,param in list(self.named_params().items()):
            g=grads[name]*scale
            self.adam_m[name]=b1*self.adam_m[name]+(1-b1)*g
            self.adam_v[name]=b2*self.adam_v[name]+(1-b2)*g*g
            mh=self.adam_m[name]/(1-b1**self.step); vh=self.adam_v[name]/(1-b2**self.step)
            self.set_param(name, param - lr*mh/(np.sqrt(vh)+ep))
    def init_ema(self):
        self.ema={k:v.copy() for k,v in self.named_params().items()}
    def update_ema(self, decay=0.999):
        if not hasattr(self,'ema'): self.init_ema()
        for k,v in self.named_params().items(): self.ema[k]=decay*self.ema[k]+(1-decay)*v
    def apply_ema(self):
        self._backup={k:v.copy() for k,v in self.named_params().items()}
        for k,v in self.ema.items(): self.set_param(k, v.copy())
    def restore_ema(self):
        for k,v in self._backup.items(): self.set_param(k, v)
    def predict(self,x):
        logits,bl=self.forward(x); return logits.argmax(1), bl.argmax(1), softmax(logits)
    def save(self,path):
        np.savez_compressed(path, **self.named_params(),
            config=np.array([self.vocab,self.d,self.max_len,self.n_classes]))
    @classmethod
    def load(cls,path):
        d=np.load(path,allow_pickle=True); vocab,dm,max_len,nc=[int(v) for v in d['config']]
        m=cls(vocab,n_classes=nc,max_len=max_len,d=dm)
        # infer blocks count from keys
        nblocks=len(set(k.split('.')[0] for k in d.files if k.startswith('block')))
        if nblocks!=len(m.blocks):
            # rebuild with correct depth (heads/ff assumed default ratios)
            m=cls(vocab,n_classes=nc,max_len=max_len,d=dm,layers=nblocks)
        for k in d.files:
            if k!='config': m.set_param(k,d[k])
        return m
    def num_params(self): return sum(v.size for v in self.named_params().values())
