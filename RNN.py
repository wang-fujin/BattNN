import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('science')

class LSTM(nn.Module):
    def __init__(self,config):
        super(LSTM,self).__init__()
        self.device = config.device
        self.config = config

        self.lstm = nn.LSTM(input_size=1,hidden_size=16,num_layers=2,batch_first=True)
        self.fc = nn.Linear(16,1)

        if config.save_model is not None:
            self.save_name = f'results/LSTM-{config.save_model}-batch_size={config.batch_size}-seq_len={config.seq_len}.pkl'
        else:
            self.save_name = None

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[100,500],gamma=0.5)

        self.mse = torch.nn.MSELoss()
        self.iter = 0

    def get_data(self,x,label):
        self.input = x.to(self.device)
        self.label = label.to(self.device)


    def predict(self,input_seq):
        '''
        :param input_seq: [batch_size,seq_len]
        :return:
        '''
        input = torch.unsqueeze(input_seq,dim=-1)
        output,(h,c) = self.lstm(input)
        output = self.fc(output)
        return torch.squeeze(output),0


    def train_one_epoch(self,print_per):
        self.optimizer.zero_grad()
        pred,_ = self.predict(self.input)

        l_mse = self.mse(pred, self.label)

        loss = l_mse
        loss.backward()

        self.optimizer.step()
        self.iter += 1

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        print(
            f"\r{self.iter} loss : {loss.item():.5e}  l_mse:{l_mse.item():.7f}, lr:{lr:.4f}",
            end = ''
        )
        if self.iter % print_per == 0:
            print("")
        return loss

    def train(self,save_model=False):
        min_loss = 10
        stop = 0

        for e in range(self.config.epoch):
            loss = self.train_one_epoch(print_per=50)
            self.scheduler.step()
            stop += 1
            if loss < min_loss:
                min_loss = loss.item()
                stop = 0
                self.best_state = {'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                   'epoch': self.iter}
            if stop > 50:
                print('\nearly stop')
                print('='*100)
                break

        self.save_model()


    def save_model(self):
        if self.save_name is not None:
            torch.save(self.best_state, self.save_name)


    def load_model(self):
        checkpoint = torch.load(self.save_name)
        self.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter = checkpoint['epoch'] + 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Battery Net for Simulate Data')
    parser.add_argument('--batch_size',default=60)
    parser.add_argument('--seq_len',default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch',default=1000)
    parser.add_argument('--lr',default=2e-2)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--save_model',default='Sim',choices=[None,'Sim','NASA','Lab'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    main
    '''
    args = get_args()
    net = LSTM(args)
    paras = count_parameters(net)
    print('the number of LSTM parameters is:',paras)
    print(net)