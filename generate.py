"""Generate from a saved model: python generate.py --model GRU --temperature 1.0"""
import argparse, json
import torch
from train import ROOT, NAMES, LanguageModel, generate

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=NAMES,default='GRU')
    p.add_argument('--seed',default='the time traveller was')
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--length',type=int,default=60)
    p.add_argument('--random-seed',type=int,default=1000)
    a=p.parse_args()
    if a.temperature<=0 or a.length<2: p.error('Temperature must be positive and length at least 2.')
    torch.set_num_threads(4)
    vocab=json.loads((ROOT/'models/vocabulary.json').read_text())
    model=LanguageModel(a.model,len(vocab))
    model.load_state_dict(torch.load(ROOT/f'models/{a.model}.pt',map_location='cpu',weights_only=True))
    result=generate(model,vocab,a.seed,a.temperature,12,a.random_seed,a.length)
    print(a.seed+' '+result['text'])
