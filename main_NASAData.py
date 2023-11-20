def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Battery Net for NASA Data')
    parser.add_argument('--V0',default=4.2)
    parser.add_argument('--x0',default=[8000,0,0])
    parser.add_argument('--dt',default=1.0)
    parser.add_argument('--VEOD',default=3.2)

    parser.add_argument('--Rp',default=1000)
    parser.add_argument('--Rs',default=0.5)
    parser.add_argument('--Csp',default=15)
    parser.add_argument('--Cs',default=500)

    parser.add_argument('--batch_size',default=30)
    parser.add_argument('--seq_len',default=60)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch',default=2000)
    parser.add_argument('--lr',default=2e-2)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--model_name', default='BattNN', choices=['BattNN', 'LSTM', 'CNN'])
    parser.add_argument('--save_model', default='NASA', choices=[None,'NASA'])



    args = parser.parse_args()
    return args

from functions import train,test,save_to_txt
from dataloader.load_NASA_data import NASAdata

def main():
    args = get_args()
    data = NASAdata(n=args.batch_size,length=args.seq_len)
    train_x,train_y = data.load_train_data()

    train(args=args,train_x=train_x,train_y=train_y,model_name=args.model_name)
    error = test(args,data_iter=data.yield_test_data,model_name=args.model_name,plot=0)
    [MAE, MAPE, MSE, RMSE] = error
    info = f'model:{args.model_name}. batch size:{args.batch_size}. seq len:{args.seq_len}.  MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
    print(info)


def grid_search():
    '''
    对NASA数据集，搜索参数Rp,Rs,Csp,Cs
    :return:
    '''
    args = get_args()
    for Rp in [1000,5000,10000,50000]:
        for Rs in [0.02,0.05,0.1,0.5]:
            for Csp in [5,15,50,100]:
                for Cs in [10,100,200,500]:
                    setattr(args, 'Rp', Rp)
                    setattr(args, 'Rs', Rs)
                    setattr(args, 'Csp', Csp)
                    setattr(args, 'Cs', Cs)
                    for i in range(3):
                        try:
                            data = NASAdata(n=args.batch_size, length=args.seq_len)
                            train_x, train_y = data.load_train_data()
                            train(args=args,train_x=train_x,train_y=train_y,model_name=args.model_name)
                            error = test(args, data_iter=data.yield_test_data, model_name=args.model_name,plot=0)
                            info = f'para: Rp={Rp}.Rs={Rs}.Csp={Csp}.Cs={Cs}. experiment={i+1}. error={error}'
                            save_to_txt('grid_search.txt', info)
                        except:
                            info_error = info = f'para: Rp={Rp}.Rs={Rs}.Csp={Csp}.Cs={Cs}. experiment={i+1}. error=None'
                            save_to_txt('grid_search.txt', info_error)
                            break


def total_main():
    args = get_args()
    for model in ['BattNN', 'LSTM', 'CNN']:
        setattr(args,'model_name',model)
        batteries = ['Dis_RW3.mat','Dis_RW4.mat','Dis_RW5.mat','Dis_RW6.mat']
        for batt in batteries:
            path = f'data/NASA11/{batt}'
            for e in range(5):
                try:
                    data = NASAdata(path=path,n=args.batch_size,length=args.seq_len)
                    train_x, train_y = data.load_train_data()
                    train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
                    error = test(args, data_iter=data.yield_test_data, model_name=args.model_name,plot=0)
                    [MAE, MAPE, MSE, RMSE] = error
                    info = f'model:{model}. battery:{batt}. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
                    print(info)
                    save_to_txt(f'NASAdata results.txt', info)

                except:
                    info = f'model:{model}. battery:{batt}. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. Error'
                    print(info)
                    save_to_txt(f'NASAdata results.txt', info)




if __name__ == '__main__':
    total_main()