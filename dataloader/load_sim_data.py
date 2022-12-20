import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('science')
import random
random.seed(2022)
np.random.seed(2022)

def load_train_data(folder='data/SimData/current=[2, 8] len=[60, 200] train',n=10,length=30):
    files = os.listdir(folder)
    random.shuffle(files)
    I = []
    V = []
    count = 0
    for i in range(len(files)):
        npy = np.load(os.path.join(folder,files[i]))
        npy_len = npy.shape[1]
        if npy_len<length:
            continue
        c = npy[0,:length]
        v = npy[1,:length]
        I.append(c)
        V.append(v)
        #plt.plot(v)
        count += 1
        if count+1 > n:
            break
    #plt.show()
    if count+1 <= n:
        print(f'符合条件的数据只有{count}条！')
    current = np.array(I,dtype=np.float64)
    voltage = np.array(V,dtype=np.float64)
    return current,voltage

def yield_test_data(folder='data/SimData/current=[2, 8] len=[60, 200] test'):
    files = os.listdir(folder)
    random.shuffle(files)
    for i in range(len(files)):
        npy = np.load(os.path.join(folder,files[i]))
        c = npy[0,:]
        v = npy[1,:]
        current = np.array(c,dtype=np.float64)
        voltage = np.array(v,dtype=np.float64)
        yield current,voltage


def main():
    train_folder = '../data/SimData/current=[2, 8] len=[60, 200] train'
    current,voltage = load_train_data(train_folder)
    print(current.shape)

    test_folder = '../data/SimData/current=[2, 8] len=[60, 200] test'
    for c,v in yield_test_data(test_folder):
        print(c.shape)
        break

if __name__ == '__main__':
    main()




