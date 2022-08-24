from imports import *
from args import *
from data import *
from networks import *

class Trainer():
    def __init__(self, args):

        self.args = args
        self.start_epoch = self.load(metrics=True)

        self.traindata, self.validdata, self.testdata = self.args.data
        self.trainloader, self.validloader, self.testloader = self.get_iterator(self.args.data)
        
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        if self.start_epoch == 0:
            self.train_loss = []
            self.train_metrics = {'accuracy': []}
            self.valid_loss = []
            self.valid_metrics = {'accuracy': []}

    def get_iterator(self, data):
        train, valid, test = data
        trainloader = DataLoader(train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(self.args.seed))
        validloader = DataLoader(valid, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(self.args.seed))
        testloader = DataLoader(test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=np.random.seed(self.args.seed))
        return trainloader, validloader, testloader

    def get_criterion(self):
        return nn.CrossEntropyLoss(weight=self.args.weights).to(self.args.device)
    
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.max_epochs, eta_min=1e-12, last_epoch=-1, verbose=False)

    def get_model(self):
        model = get_model(self.args)
        return model

    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters())/1e6

    def save(self, epoch):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.args.checkpoint, f"model_{args.experiment}.pth"))
        torch.save({
            'epoch': epoch,
            'args': args,
            'loss': (self.train_loss, self.valid_loss),
            'metrics': (self.train_metrics, self.valid_metrics)
            }, os.path.join(self.args.checkpoint, f"metrics_{args.experiment}.pth"))
        
    def load(self, metrics=False):
        if not metrics:
            if os.path.exists(os.path.join(self.args.checkpoint, f"model_{args.experiment}.pth")):
                checkpoints = torch.load(os.path.join(self.args.checkpoint, f"model_{args.experiment}.pth"), map_location=self.args.device)
                self.model.load_state_dict(checkpoints['model_state_dict'])
                self.optimizer.load_state_dict( checkpoints['optimizer_state_dict'])

        if os.path.exists(os.path.join(self.args.checkpoint, f"metrics_{args.experiment}.pth")):
            checkpoints = torch.load(os.path.join(self.args.checkpoint, f"metrics_{args.experiment}.pth"), map_location=self.args.device)
            self.args = checkpoints['args']
            self.train_loss, self.valid_loss = checkpoints['loss']
            self.train_metrics, self.valid_metrics = checkpoints['metrics']
            return checkpoints['epoch']
        return 0

    def train(self):
        epoch_loss = 0
        epoch_metrics = {'accuracy': 0}

        torch.cuda.empty_cache()
        self.model.train()

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.trainloader), bar_char='█')
            for index, (data, label) in enumerate(self.trainloader):
                data = data.to(self.args.device).float()
                label = label.long().to(self.args.device)

                self.optimizer.zero_grad()
                
                output = self.model(data)

                loss = self.criterion(output, label)

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()/len(self.trainloader)

                epoch_metrics['accuracy'] += (torch.argmax(output, dim=1)==label).float().sum().item()/len(self.traindata)

                bar.update()
                torch.cuda.empty_cache()

        return epoch_loss, epoch_metrics

    def evaluate(self):
        epoch_loss = 0
        epoch_metrics = {'accuracy': 0}

        torch.cuda.empty_cache()
        self.model.eval()

        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                bar = pyprind.ProgBar(len(self.validloader), bar_char='█')
                for index, (data, label) in enumerate(self.validloader):
                    data = data.to(self.args.device).float()
                    label = label.long().to(self.args.device)

                    output = self.model(data)

                    loss = self.criterion(output, label)

                    epoch_loss += loss.item()/len(self.validloader)
                    epoch_metrics['accuracy'] += (torch.argmax(output, dim=1)==label).float().sum().item()/len(self.validdata)

                    bar.update()
                    torch.cuda.empty_cache()

        return epoch_loss, epoch_metrics

    def test(self):

        torch.cuda.empty_cache()
        self.model.eval()

        outputs = torch.empty([0,])

        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                bar = pyprind.ProgBar(len(self.testloader), bar_char='█')
                for index, (data, label) in enumerate(self.testloader):
                    data = data.to(self.args.device)

                    output = torch.argmax(self.model(data)).detach().cpu()
                    outputs = torch.cat((outputs, output), dim=0)

                    bar.update()
                    torch.cuda.empty_cache()

        return outputs
    
    def fit(self, next=True):
        if next:
            self.start_epoch = self.load()

        for epoch in range(self.start_epoch+1, self.args.max_epochs+1, 1):

            epoch_train_loss, epoch_train_metrics = self.train()
            epoch_train_accuracy = epoch_train_metrics['accuracy']

            self.train_loss.append(epoch_train_loss)
            self.train_metrics['accuracy'].append(epoch_train_metrics['accuracy'])

            epoch_valid_loss, epoch_valid_metrics = self.evaluate()
            epoch_valid_accuracy = epoch_valid_metrics['accuracy']

            
            self.valid_loss.append(epoch_valid_loss)
            self.valid_metrics['accuracy'].append(epoch_valid_metrics['accuracy']) 

            # self.scheduler.step()
            if epoch_valid_metrics['accuracy'] >= max(self.valid_metrics['accuracy']):
                self.save(epoch)

            time.sleep(1)
            print(f'Epoch {epoch}/{self.args.max_epochs} | Training: Loss = {round(epoch_train_loss, 4)}  Accuracy = {round(epoch_train_accuracy, 4)} | Validation: Loss = {round(epoch_valid_loss, 4)}  Accuracy = {round(epoch_valid_accuracy, 4)}')
            
if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.data = data
    args.weights = weights

    trainer = Trainer(args)