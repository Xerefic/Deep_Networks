import os
import numpy as np
import torch
import itertools
from typing import Union, List, Dict, Any, Optional, cast

from args import *
from data import *
from dnn import *
from dgn import *
from dlgn import *
from dlgnsf import *
from trainer import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def complement(x):
    return torch.ones_like(x) - x

def process_AND(x):
    if len(x) > 1:
        t = process_AND(x[1:])
        out = [x[0]*e for e in t]
        out.extend([complement(x[0])*e for e in t])
        return out
    elif len(x) == 1:
        return [x[0], complement(x[0])]

def process_OR(x):
    def or_gate(x):
        if len(x) > 2:
            y = [or_gate(x[0:2])]
            y.extend(x[2:])
            return or_gate(y)
        if len(x) == 2:
            t = torch.cat([y.unsqueeze(0) for y in x], dim=0)
            t = torch.prod(torch.ones_like(t) - t, dim=0)
            t = torch.ones_like(t) - t
            return t
        if len(x) == 1:
            return x[0]
    out = or_gate(x)
    return out

def gate(x, exposure):
    if len(x) > 2:
        t = process_AND(x)
        t = list(itertools.compress(t, exposure))
        t = process_OR(t)
        return t
    else:
        return x[0]


def get_models(args, mode='Same'):
    set_seed(args.seed)
    
    models = {
    'DNN': DNN(args),
    'DGN': DGN(args),
    'DLGN': DLGN(args),
    'DLGNSF': DLGNSF(args),
    }
    # if os.path.exists(os.path.join("/content", "model.pth")):
    #     checkpoints = torch.load(os.path.join("/content", "model.pth"), map_location=args.device)
    #     models['DNN'].load_state_dict(checkpoints['model_state_dict'])

    if mode=='Same':
        for model in ['DGN', 'DLGN', 'DLGNSF']:
            architecture = models[model]
            parent = dict(models['DNN'].named_children())
            if model!='DLGNSF':         
                for fc in architecture.named_children():
                    if isinstance(fc[1], torch.nn.ModuleList):
                        for m in dict(architecture.named_children())[fc[0]]:
                            m.weight.data = parent[fc[0][:3]].weight.data.clone()
                            m.bias.data = parent[fc[0][:3]].bias.data.clone()
            else:
                for fc in architecture.named_children():
                    if isinstance(fc[1], torch.nn.ModuleList):
                        for m in dict(architecture.named_children())[fc[0]]:
                            m.weight.data = parent['fc0'].weight.data.clone()
                            m.bias.data = parent['fc0'].bias.data.clone()

    return models

def get_trainer(args, model='DNN', mode='Same'):
    architecture = get_models(args)[model]
    trainer = Trainer(args, architecture)
    return trainer

def get_plot(valid_metrics, name):
    color = []
    for i in range(len(valid_metrics['subnetwork'])):
        color.append('#%06X' % random.randint(0, 0xFFFFFF))

    fig,ax = plt.subplots(frameon=False)

    for i, key in enumerate(valid_metrics['subnetwork'].keys()):
        data = valid_metrics['subnetwork'][key]
        p = poly(data, color=color[i])
        ax.add_patch(p)

    ax.set_xlim([-args.range,args.range])
    ax.set_ylim([-args.range,args.range])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0)