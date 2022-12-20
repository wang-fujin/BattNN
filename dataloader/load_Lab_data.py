import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import os
random.seed(2022)
np.random.seed(2022)

class LabData():
    def __init__(self,path='data/LabData',n=50,length=50):
        self.root = path
        super(LabData,self).__init__()
        self.files = os.listdir(path)
        random.shuffle(self.files)
        self.train_files = self.files[:n]
        self.test_files = self.files[n:]
        self.length = length

    def load_train_data(self):
        I, V = [], []
        for file in self.train_files:
            data = np.load(os.path.join(self.root,file))
            current = data[0, :self.length]
            voltage = data[1, :self.length]
            I.append(current)
            V.append(voltage)

        current = np.array(I, dtype=np.float32)
        voltage = np.array(V, dtype=np.float32)
        return current,voltage

    def yield_test_data(self):
        for file in self.test_files:
            data = np.load(os.path.join(self.root, file))
            current = data[0,:]
            voltage = data[1,:]
            yield current,voltage

class LabDataSperate():
    def __init__(self,path='data/LabData',n=50,length=50,test_id=1):
        super(LabDataSperate,self).__init__()
        self.root = path
        self.files = os.listdir(path)
        test_battery = f'B{test_id}'

        self.train_files = []
        self.test_files = []
        for file in self.files:
            if test_battery in file:
                self.test_files.append(file)
            else:
                self.train_files.append(file)
        random.shuffle(self.train_files)

        self.train_files = self.train_files[:n]
        self.length = length

    def load_train_data(self):
        I, V = [], []
        for file in self.train_files:
            data = np.load(os.path.join(self.root,file))
            current = data[0, :self.length]
            voltage = data[1, :self.length]
            I.append(current)
            V.append(voltage)

        current = np.array(I, dtype=np.float32)
        voltage = np.array(V, dtype=np.float32)
        return current,voltage

    def yield_test_data(self):
        for file in self.test_files:
            data = np.load(os.path.join(self.root, file))
            current = data[0,:]
            voltage = data[1,:]
            yield current,voltage

class LabDataFailure():
    def __init__(self,test_id=1,n=50,length=50):
        super(LabDataFailure, self).__init__()
        self.test_id = test_id
        self.length = length
        train_path = 'data/LabData'
        test_path = 'data/LabData_failure'
        self.files = os.listdir(train_path)

        self.train_files =self.files[40:]
        random.shuffle(self.train_files)
        self.train_files = self.train_files[:n]

        self.test_files = os.listdir(test_path)
        self.test_files1 = self.test_files[:20]
        self.test_files2 = self.test_files[20:]

    def load_train_data(self):
        I, V = [], []
        for file in self.train_files:
            data = np.load(os.path.join('data/LabData',file))
            current = data[0, :self.length]
            voltage = data[1, :self.length]
            I.append(current)
            V.append(voltage)

        current = np.array(I, dtype=np.float32)
        voltage = np.array(V, dtype=np.float32)
        return current,voltage

    def yield_test_data(self):
        if self.test_id == 1:
            files = self.test_files1
        else:
            files = self.test_files2

        for file in files:
            data = np.load(os.path.join('data/LabData_failure', file))
            current = data[0,:]
            voltage = data[1,:]
            yield current,voltage



def plot_current_and_voltage_twin(current,voltage,title=None):
    fig, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()
    ax1.plot(list(range(len(current))), current, color='#056db1',linewidth=2) ##91c860
    ax1.set_ylabel('Current(A)', color='#056db1')
    ax1.set_xlabel('Sample points')
    ax1.tick_params(axis='y', labelcolor='#056db1')

    ax2.plot(list(range(len(voltage))), voltage, color='#fe0000',linewidth=2)
    ax2.set_ylabel('Voltage(V)', color='#fe0000')
    ax2.tick_params(axis='y', labelcolor='#fe0000')

    #title = f'len:{len(voltage)}'
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    path = '../data/LabData'
    data = LabData(path=path,length=100)
    current,voltage = data.load_train_data()
    plot_current_and_voltage_twin(current[0],voltage[0])



if __name__ == '__main__':
    main()
