"""Supplementary known-word evaluation and a smoothed unigram baseline."""
import collections, json, math, re
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import ROOT, NAMES, LanguageModel, tokenize

def main():
    torch.set_num_threads(4)
    raw=(ROOT/'data/time_machine.txt').read_text(encoding='utf-8-sig')
    body=re.split(r'\*\*\* START OF .*?\*\*\*',raw,maxsplit=1)[1]
    body=re.split(r'\*\*\* END OF .*?\*\*\*',body,maxsplit=1)[0]
    words=tokenize(body[body.lower().find('the time traveller (for so it will be convenient'):])
    vocab=json.loads((ROOT/'models/vocabulary.json').read_text()); ids={w:i for i,w in enumerate(vocab)}
    train=[ids.get(w,0) for w in words[:int(len(words)*.8)]]
    test=torch.tensor([ids.get(w,0) for w in words[int(len(words)*.9):]])
    windows=test.unfold(0,13,1); x,y=windows[:,:12],windows[:,12]
    counts=collections.Counter(train); probs=torch.tensor([(counts[i]+1)/(len(train)+len(vocab)) for i in range(len(vocab))])
    loss=-probs[y].log().mean().item()
    out={'unigram':{'test_loss':loss,'test_perplexity':math.exp(loss),'test_accuracy':(y==probs.argmax()).float().mean().item()},'models':{}}
    for name in NAMES:
        model=LanguageModel(name,len(vocab)); model.load_state_dict(torch.load(ROOT/f'models/{name}.pt',weights_only=True)); model.eval()
        correct=known=0
        with torch.no_grad():
            for bx,by in DataLoader(TensorDataset(x,y),batch_size=256):
                mask=by!=0; correct+=((model(bx).argmax(1)==by)&mask).sum().item(); known+=mask.sum().item()
        out['models'][name]={'known_target_accuracy':correct/known,'known_targets':known}
    (ROOT/'results/supplementary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
