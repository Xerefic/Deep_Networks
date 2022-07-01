from imports import *
from args import *

# def process():
#     data = pd.read_csv(os.path.join(args.root_dir, 'train_labels.csv'))
#     encoder = preprocessing.LabelEncoder()
#     encoder.fit(data['Label'])
#     return encoder

def get_data(args):
    # encoder = process()

    data = pd.read_csv(os.path.join(args.root_dir, 'train_file.csv'))
    files = data['Image']
    labels = data['Label']
    train_data_files = files
    train_data_labels = labels
    # train_data_files, valid_data_files, train_data_label, valid_data_label = train_test_split(files, labels, test_size=0.05, shuffle=True, stratify=labels)
    
    data = pd.read_csv(os.path.join(args.root_dir, 'test_file.csv'))
    test_data_files = data['Image']
    test_data_labels = data['Label']

    train_data = pd.DataFrame({'id': train_data_files, 'label': train_data_labels})
    valid_data = pd.DataFrame({'id': test_data_files, 'label': test_data_labels})
    test_data = pd.DataFrame({'id': test_data_files, 'label': test_data_labels})

    unique, counts = np.unique(labels, return_counts=True)
    weights = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=unique, y=np.asarray(labels)))

    return (train_data, valid_data, test_data), weights


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        if mode=='train':
            self.entry = self.args.train_file
        elif mode=='valid':
            self.entry = self.args.valid_file
            self.mode = 'train'
        elif mode=='test':
            self.entry = self.args.test_file    

        self.transform = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def __getitem__(self, index):
        data = self.entry.iloc[index]
        file_name = os.path.join(self.args.root_dir, self.mode, str(data['id']))
        image = Image.open(file_name)
        image = self.transform(image)
        label = data['label']
        return image, label

    def __len__(self):
        return len(self.entry)

if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.train_file = data[0]
    args.valid_file = data[1]
    args.test_file = data[2]

    train_dataset = CreateDataset(args, mode='train')
    valid_dataset = CreateDataset(args, mode='valid')
    test_dataset = CreateDataset(args, mode='test')

    args.data = (train_dataset, valid_dataset, test_dataset)

    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))