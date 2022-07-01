from utils import *
from args import *
from models import *

def get_models(args, mode='Same'):
    set_seed(args.seed)
    
    models = {
    'DNN': DNN(args),
    'DGN': DGN(args),
    'DLGN': DLGN(args),
    'DLGNSF': DLGNSF(args)
    }

    if mode=='Same':
        for model in ['DGN', 'DLGN']:

            parent = dict(models['DNN'].named_children())
            if model!='DLGNSF':         
                for layer in models[model].named_children():
                    if isinstance(layer[1], torch.nn.ModuleList):
                        for m in dict(models[model].named_children())[layer[0]]:
                            m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                            m.bias.data = parent[layer[0][:-1]].bias.data.clone()
                    elif isinstance(layer[1], torch.nn.Conv2d):
                        m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                        m.bias.data = parent[layer[0][:-1]].bias.data.clone()
                    elif isinstance(layer[1], torch.nn.Linear):
                        m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                        m.bias.data = parent[layer[0][:-1]].bias.data.clone()
            elif model=='DLGNSF':
                for layer in models[model].named_children():
                    if isinstance(layer[1], torch.nn.ModuleList):
                        for m in dict(models[model].named_children())[layer[0]]:
                            m.weight.data = parent['conv0'].weight.data.clone()
                            m.bias.data = parent['conv0'].bias.data.clone()
                    elif isinstance(layer[1], torch.nn.Conv2d):
                        m.weight.data = parent['conv0'].weight.data.clone()
                        m.bias.data = parent['conv0'].bias.data.clone()
                    elif isinstance(layer[1], torch.nn.Linear):
                        m.weight.data = parent['fc5'].weight.data.clone()
                        m.bias.data = parent['fc5'].bias.data.clone()

    return models

if __name__ == "__main__":
    args = TrainingArgs()
    models = get_models(args, mode='Same')