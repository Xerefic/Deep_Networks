from utils import *
from args import *
from models import *

def get_model(args):
    def initialize_weights(model):
        parent = dict(dnn.named_children())
        if args.architecture!='DLGNSF':         
            for layer in model.named_children():
                if isinstance(layer[1], torch.nn.ModuleList):
                    for m in dict(model.named_children())[layer[0]]:
                        m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                        m.bias.data = parent[layer[0][:-1]].bias.data.clone()
                elif isinstance(layer[1], torch.nn.Conv2d):
                    m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                    m.bias.data = parent[layer[0][:-1]].bias.data.clone()
                elif isinstance(layer[1], torch.nn.Linear):
                    m.weight.data = parent[layer[0][:-1]].weight.data.clone()
                    m.bias.data = parent[layer[0][:-1]].bias.data.clone()
        elif args.architecture=='DLGNSF':
            for layer in model.named_children():
                if isinstance(layer[1], torch.nn.ModuleList):
                    for m in dict(model.named_children())[layer[0]]:
                        m.weight.data = parent['conv0'].weight.data.clone()
                        m.bias.data = parent['conv0'].bias.data.clone()
                elif isinstance(layer[1], torch.nn.Conv2d):
                    m.weight.data = parent['conv0'].weight.data.clone()
                    m.bias.data = parent['conv0'].bias.data.clone()
                elif isinstance(layer[1], torch.nn.Linear):
                    m.weight.data = parent['fc5'].weight.data.clone()
                    m.bias.data = parent['fc5'].bias.data.clone()
        return model

    set_seed(args.seed)
    dnn = DNN(args).to(args.device)
    if args.architecture=='DNN':
        return dnn
    elif args.architecture=='DGN':
        model = DGN(args).to(args.device)
    elif args.architecture=='DLGN':
        model = DLGN(args).to(args.device)
    elif args.architecture=='DLGNSF':
        model = DLGNSF(args).to(args.device)

    if args.mode=='Same':
        return initialize_weights(model)
    elif args.mode=='Random':
        return model

if __name__ == "__main__":
    args = TrainingArgs()
    model = get_model(args)