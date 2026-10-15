"""Reproducible word-level recurrent language model comparison. Run from any directory."""
import argparse, collections, csv, hashlib, json, math, platform, random, re, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = ['SimpleRNN', 'BidirectionalRNN', 'LSTM', 'GRU']
SEEDS = ['the time traveller was', 'i looked around me', 'the machine was standing']
def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace('\u2019', "'"))

class LanguageModel(nn.Module):
    def __init__(self, name, vocab_size):
        super().__init__()
        self.name = name
        self.embedding = nn.Embedding(vocab_size, 64)
        cls = {'SimpleRNN':nn.RNN, 'BidirectionalRNN':nn.RNN, 'LSTM':nn.LSTM, 'GRU':nn.GRU}[name]
        self.recurrent = cls(64, 96, batch_first=True, bidirectional=name=='BidirectionalRNN')
        self.output = nn.Linear(192 if name=='BidirectionalRNN' else 96, vocab_size)
    def forward(self, x):
        _, h = self.recurrent(self.embedding(x))
        if isinstance(h, tuple): h = h[0]
        h = torch.cat([h[-2], h[-1]], dim=1) if self.name=='BidirectionalRNN' else h[-1]
        return self.output(h)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); loss=correct=n=0
    for x,y in loader:
        logits=model(x); loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return loss/n, correct/n

@torch.no_grad()
def generate(model, vocab, seed, temperature, context, rng_seed, length=60):
    model.eval(); ids={w:i for i,w in enumerate(vocab)}
    tokens=tokenize(seed); emitted=[]; g=torch.Generator().manual_seed(rng_seed)
    for _ in range(length):
        x=torch.tensor([[ids.get(w,0) for w in tokens[-context:]]])
        logits=model(x)[0]/temperature; logits[0]=-torch.inf
        word=vocab[torch.multinomial(logits.softmax(0),1,generator=g).item()]
        tokens.append(word); emitted.append(word)
    bigrams=list(zip(emitted,emitted[1:]))
    return {'text':' '.join(emitted),'distinct1':len(set(emitted))/len(emitted),
            'distinct2':len(set(bigrams))/len(bigrams),'repeated_bigram_fraction':1-len(set(bigrams))/len(bigrams)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads); torch.use_deterministic_algorithms(True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    for folder in ['results','models']: (ROOT/folder).mkdir(exist_ok=True)
    raw=(ROOT/'data/time_machine.txt').read_text(encoding='utf-8-sig')
    body=re.split(r'\*\*\* START OF .*?\*\*\*',raw,maxsplit=1)[1]
    body=re.split(r'\*\*\* END OF .*?\*\*\*',body,maxsplit=1)[0]
    # Remove the title and contents by starting at the first narrative sentence.
    start=body.lower().find('the time traveller (for so it will be convenient')
    assert start>=0, 'Narrative start not found; check dataset edition.'
    tokens=tokenize(body[start:]); n=len(tokens); cut1=int(n*.8); cut2=int(n*.9)
    splits=[tokens[:cut1],tokens[cut1:cut2],tokens[cut2:]]
    counts=collections.Counter(splits[0]); vocab=['<unk>']+sorted(w for w,c in counts.items() if c>=2)
    ids={w:i for i,w in enumerate(vocab)}; context=12; datasets=[]
    for s in splits:
        encoded=np.array([ids.get(w,0) for w in s],dtype=np.int64)
        windows=np.lib.stride_tricks.sliding_window_view(encoded,context+1).copy()
        datasets.append(TensorDataset(torch.from_numpy(windows[:,:-1]),torch.from_numpy(windows[:,-1])))
    stats={'source':'https://www.gutenberg.org/ebooks/35','download':'https://www.gutenberg.org/ebooks/35.txt.utf-8',
           'sha256':hashlib.sha256((ROOT/'data/time_machine.txt').read_bytes()).hexdigest(),
           'tokens':n,'raw_unique_words':len(set(tokens)),'vocabulary_size':len(vocab),'context':context,
           'split_tokens':[len(s) for s in splits],'split_examples':[len(d) for d in datasets],
           'oov_rates':[sum(w not in ids for w in s)/len(s) for s in splits],
           'epochs':a.epochs,'batch_size':128,'learning_rate':.002,'seed':42,
           'embedding_size':64,'hidden_size_per_direction':96,'threads':a.threads,
           'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform(),
           'device':'CPU','processor':platform.processor()}
    (ROOT/'results/config.json').write_text(json.dumps(stats,indent=2))
    (ROOT/'models/vocabulary.json').write_text(json.dumps(vocab))
    print(json.dumps(stats),flush=True)
    valid=DataLoader(datasets[1],batch_size=256); test=DataLoader(datasets[2],batch_size=256)
    rows=[]; histories={}; samples=[]
    for name in NAMES:
        torch.manual_seed(42)
        loader=DataLoader(datasets[0],batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42))
        model=LanguageModel(name,len(vocab)); opt=torch.optim.Adam(model.parameters(),lr=.002)
        best=float('inf'); history=[]; started=time.perf_counter()
        for epoch in range(1,a.epochs+1):
            model.train(); total=ntrain=0; tick=time.perf_counter()
            for x,y in loader:
                opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                total+=loss.item()*len(y); ntrain+=len(y)
            vl,va=evaluate(model,valid)
            history.append({'epoch':epoch,'train_loss':total/ntrain,'val_loss':vl,'val_accuracy':va,'seconds':time.perf_counter()-tick})
            if vl<best:
                best=vl; best_epoch=epoch; torch.save(model.state_dict(),ROOT/f'models/{name}.pt')
            print(f'{name} {epoch}/{a.epochs} train={total/ntrain:.3f} val={vl:.3f}',flush=True)
        elapsed=time.perf_counter()-started
        model.load_state_dict(torch.load(ROOT/f'models/{name}.pt',weights_only=True)); tl,ta=evaluate(model,test)
        rows.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
                     'validation_loss':best,'test_loss':tl,'test_perplexity':math.exp(tl),'test_accuracy':ta,'training_seconds':elapsed})
        histories[name]=history
        for seed_idx,seed in enumerate(SEEDS):
            for temperature in [.5,1.,1.5]:
                for repeat in range(3):
                    samples.append({'model':name,'seed':seed,'temperature':temperature,'replicate':repeat,
                                    **generate(model,vocab,seed,temperature,context,1000+seed_idx*10+repeat)})
        (ROOT/'results/metrics.json').write_text(json.dumps(rows,indent=2))
        (ROOT/'results/history.json').write_text(json.dumps(histories,indent=2))
        (ROOT/'results/samples.json').write_text(json.dumps(samples,indent=2))
    with (ROOT/'results/metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,h in histories.items():
        ax[0].plot([v['epoch'] for v in h],[v['train_loss'] for v in h],label=name)
        ax[1].plot([v['epoch'] for v in h],[v['val_loss'] for v in h],label=name)
    for i,title in enumerate(['Training loss','Validation loss']):
        ax[i].set(title=title,xlabel='Epoch',ylabel='Cross entropy'); ax[i].legend(fontsize=8); ax[i].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT/'results/loss_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar([r['model'] for r in rows],[r['test_perplexity'] for r in rows],color=['#334155','#64748b','#0891b2','#0f766e'])
    ax.set(ylabel='Test perplexity',title='Held-out next-word prediction'); fig.tight_layout(); fig.savefig(ROOT/'results/perplexity.png',dpi=180); plt.close(fig)
    print('COMPLETE',flush=True)
if __name__=='__main__': main()
