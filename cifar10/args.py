from imports import *

exposures = {
    1: ([1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]),
    2: ([1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]),
    3: ([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    4: ([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    5: ([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
}

def get_gates(gates, coefficient, image_size, channels):
    def get_mixing(image_size, channel_size):
        def get_layer(image_size):
            total = image_size**2
            pure = int(coefficient*total)
            random = total - pure
            t = torch.cat((torch.ones((pure,)), torch.zeros((random,))), dim=0)
            idx = torch.randperm(t.nelement())
            t = t.view(-1)[idx].view(1, image_size, image_size)
            return t
        return torch.cat([get_layer(image_size) for _ in range(channel_size)], dim=0)
    gating = []
    for i, gate in enumerate(gates):
        gating.append((gate, get_mixing(image_size, channels[i])))
    return tuple(gating)

@dataclass
class TrainingArgs():

    seed: int = 1
    lr: float = 3e-4
    batch_size: int = 32
    num_workers: int = os.cpu_count()
    max_epochs: str = 200
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size: int = int(1e4)
    bound: int = 10
    input_size: int = 3
    output_size: int = 10
    image_size: int = 32
    channels: tuple = (16, 16, 16, 16, 16)
    k: int = 1
    beta: int = 1
    weights: torch.Tensor = None
    gating: tuple = get_gates(['AND', 'AND', 'AND', 'AND', 'AND'], 0.8, image_size, channels)
    exposure: tuple = tuple(torch.where(torch.rand(len(channels), 2**k)>0.5, torch.ones(1,), torch.zeros(1,)).tolist())

    train_file: tuple = None
    valid_file: tuple = None
    test_file: tuple = None
    data: tuple = None

    root_dir: str = './data/cifar-10'
    checkpoint: str = './checkpoints'
    experiment: str = None

args = TrainingArgs()