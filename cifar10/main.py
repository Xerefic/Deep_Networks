from imports import *
from args import *
from utils import *
from data import *
from networks import *
from trainer import *

if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.data = data
    args.weights = weights

    experiment = {}
    if os.path.exists(os.path.join(args.checkpoint, 'experiments.pkl')):
        with open(os.path.join(args.checkpoint, 'experiments.pkl'), 'rb') as handle:
            experiment = pkl.load(handle)

    models = ['DNN', 'DGN', 'DLGN', 'DLGNSF'] # 
    seeds = [420, 999, 785, 2565, 3821, 856, 9999, 1001, 565, 7890] #
    modes = ['Random', 'Same'] # 
    ks = range(1, 5+1, 1) #
    logics = ['ALL', 'AND', 'OR']


    for logic in logics:
        for mode in modes:
            for seed in seeds:
                for model in models:
                    for k in ks:
                        exp = f'k{k}{model}_{mode}_{logic}'
                        name = f'k{k}{model}_{mode}_{logic}_{seed}'

                    
                        args.seed = seed
                        args.k = k
                        args.experiment = name
                        args.gating = get_gates([logic for _ in range(len(args.channels))], 0.8, args.image_size, args.channels)
                        # args.exposure = exposures[args.k]
                        args.exposure = tuple(torch.where(torch.rand(len(args.channels), 2**args.k)>0.5, torch.ones(1,), torch.zeros(1,)).tolist())

                        trainer = get_trainer(args, model, mode)
                        print(f'Model: {name}')
                        trainer.fit()

                        accuracy = max(trainer.valid_metrics['accuracy'])
                        print(f'Model: {name} | Accuracy: {accuracy}')

                        if exp in experiment.keys():
                            experiment[exp].append(accuracy)
                        else:
                            experiment[exp] = [accuracy]

                        with open('experiments.pkl', 'wb') as handle:
                            pkl.dump(experiment, handle, protocol=pkl.HIGHEST_PROTOCOL)