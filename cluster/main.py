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

    architectures = ['DLGN'] # 'DNN', 'DGN', 'DLGN', 'DLGNSF'
    seeds = [420, 999, 785, 2565, 3821] # 420, 999, 785, 2565, 3821, 856, 9999, 1001, 565, 7890
    modes = ['Random'] # 'Random', 'Same'
    ks = [3] # range(1, 5+1, 1)
    logics = [['AND', 'AND', 'AND', 'AND', 'AND']] # 'MIX', 'ALL', 'AND', 'OR'
    mixing = (0.5, 0.5, 0.5, 0.5, 0.5)

    for logic in logics:
        for mode in modes:
            for seed in seeds:
                for architecture in architectures:
                    for k in ks:
                        logic_ = '_'.join(logic)
                        exp = f'k{k}{architecture}_{mode}_{logic_}'
                        name = f'k{k}{architecture}_{mode}_{logic_}_{seed}'

                    
                        args.seed = seed
                        args.k = k
                        args.experiment = name
                        args.architecture = architecture
                        args.mode = mode
                        args.gate = logic
                        args.mixing = mixing
                        args.gating = get_gates(args.gate, args.mixing, args.image_size, args.channels)
                        args.exposure = get_exposure(len(args.channels), k)

                        trainer = Trainer(args)
                        print(f'Model: {name}')
                        trainer.fit()

                        accuracy = max(trainer.valid_metrics['accuracy'])
                        print(f'Model: {name} | Accuracy: {accuracy}')

                        if exp in experiment.keys():
                            experiment[exp].append(accuracy)
                        else:
                            experiment[exp] = [accuracy]

                        with open(os.path.join(args.checkpoint, 'experiments.pkl'), 'wb') as handle:
                            pkl.dump(experiment, handle, protocol=pkl.HIGHEST_PROTOCOL)