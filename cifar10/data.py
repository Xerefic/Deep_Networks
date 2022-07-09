from imports import *
from args import *

def get_data(args):

    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    train_dataset = torchvision.datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=transform)

    unique, counts = np.unique(train_dataset.targets, return_counts=True)
    weights = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=unique, y=np.asarray(train_dataset.targets)))

    return (train_dataset, valid_dataset, test_dataset), weights

if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.data = data

    print(weights)
    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))

    # trainloader = DataLoader(data[0], batch_size=32, shuffle=True)
    # for i, data in enumerate(trainloader):
    #     continue
    # validloader = DataLoader(data[1], batch_size=32, shuffle=False)
    # for i, data in enumerate(validloader):
    #     continue
    # testloader = DataLoader(data[2], batch_size=32, shuffle=False)
    # for i, data in enumerate(testloader):
    #     continue
