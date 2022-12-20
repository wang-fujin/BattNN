
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
random.seed(2022)
np.random.seed(2022)



class NASAdata():
    def __init__(self,path='data/NASA11/Dis_RW3.mat',n=50,length=50):
        super(NASAdata,self).__init__()
        self.data = loadmat(path)['re_discharge_data']
        self.random_index = np.random.permutation(self.data.shape[1])
        self.train_index = self.random_index[:n]
        self.test_index = self.random_index[n:]
        self.length = length

    def load_train_data(self):
        I, V = [], []
        for idx in self.train_index:
            current = self.data[0, idx][0][0, :self.length]
            voltage = self.data[0, idx][1][0, :self.length]
            if current is None:
                continue
            I.append(current)
            V.append(voltage)
        current = np.array(I, dtype=np.float32)
        voltage = np.array(V, dtype=np.float32)
        return current,voltage

    def yield_test_data(self):
        for i in self.test_index:
            current = self.data[0, i][0][0,:]
            voltage = self.data[0, i][1][0,:]
            if current is not None:
                yield current,voltage

def look_up_data():
    plt.style.use('science')
    path = '../data/NASA11/Dis_RW3.mat'
    NASA = NASAdata(path=path)
    current, voltage = NASA.load_train_data()
    for c,v in NASA.yield_test_data():
        fig, ax1 = plt.subplots(1, 1)
        ax2 = ax1.twinx()
        print(c.shape)
        ax1.plot(c, color='g')
        ax1.set_ylabel('Current(A)', color='g')
        ax1.set_xlabel('Time(s)')
        ax1.tick_params(axis='y', labelcolor='g')

        ax2.plot(v, color='r')
        ax2.set_ylabel('Voltage(V)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        #plt.savefig('1.png',format='png')
        #plt.show()
        break

def plot_all_data(path='../data/NASA11/Dis_RW3.mat'):
    plt.style.use('science')
    width = 8
    height = 4
    figure_size_cm = (width / 2.54, height / 2.54)

    NASA = NASAdata(path=path)
    fig = plt.figure(figsize=figure_size_cm,dpi=150)
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    for c, v in NASA.yield_test_data():
        ax1.plot(c, color='g')
        ax1.set_ylabel('Current(A)', color='g')
        ax1.set_xlabel('Time(s)')
        ax1.tick_params(axis='y', labelcolor='g')

        ax2.plot(v, color='r')
        ax2.set_ylabel('Voltage(V)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    plt.show()



def main():
    path = '../data/NASA11/Dis_RW3.mat'
    NASA = NASAdata(path=path)
    current,voltage = NASA.load_train_data()
    print(current.shape)

    for c,v in NASA.yield_test_data():
        plt.plot(c)
        plt.show()
        plt.plot(v)
        plt.show()
        break


if __name__ == '__main__':
    plot_all_data()

