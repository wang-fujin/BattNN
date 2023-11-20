import torch
import numpy as np
from sklearn import metrics
from BattNN import BattNN
from RNN import LSTM
from CNN import CNN
import matplotlib.pyplot as plt
plt.style.use(['science','ieee'])

def eval_metrix(true_label, pred_label):
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label)
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label, pred_label))
    return [MAE, MAPE, MSE, RMSE]

def test(args,data_iter,model_name='BattNN',plot=5):

    if model_name == 'BattNN':
        model = BattNN(args)
    elif model_name == 'LSTM':
        model = LSTM(args)
    elif model_name == 'CNN':
        model = CNN(args)
    else:
        raise
    model.load_model()
    count = 0
    ERROR = []
    for c, v in data_iter():
        count += 1
        c_tensor, v_tensor = torch.from_numpy(c.astype(np.float32)), torch.from_numpy(v.astype(np.float32))

        pred, _ = model.predict(c_tensor.view(1,-1))
        pred = pred.detach().cpu().numpy()
        if model_name == 'BattNN':
            pred = pred[0]
        metrix = eval_metrix(v,pred)
        ERROR.append(metrix)

        if count <= plot:
            plt.plot(v.reshape(-1), '-.', label='label')
            plt.plot(pred, '-.', label='pred')
            plt.title(f'Test, len={v.shape[0]}, mse={metrix[2]:.4f}')
            plt.legend()
            plt.show()

    error = np.array(ERROR)
    mean_error = np.mean(error,axis=0)
    print('test error:',mean_error)
    return mean_error

def train(args,train_x,train_y,model_name='BattNN'):
    if model_name == 'BattNN':
        model = BattNN(args)
        print('select model: BattNN')
    elif model_name == 'LSTM':
        model = LSTM(args)
        print('select model: LSTM')
    elif model_name == 'CNN':
        model = CNN(args)
        print('select model: CNN')
    else:
        raise
    c_tensor, v_tensor = torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32))
    model.get_data(x=c_tensor, label=v_tensor)
    model.train()

def save_to_txt(save_name,string):
    f = open(save_name,mode='a')
    f.write(string)
    f.write('\n')
    f.close()