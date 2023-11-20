def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Battery Net for Lab Data')
    parser.add_argument('--V0',default=4.2)
    parser.add_argument('--x0',default=[8000,0,0])
    parser.add_argument('--dt',default=1.0)
    parser.add_argument('--VEOD',default=3.2)

    parser.add_argument('--Rp',default=6000)
    parser.add_argument('--Rs',default=1.0)
    parser.add_argument('--Csp',default=40)
    parser.add_argument('--Cs',default=800)

    parser.add_argument('--batch_size',default=30)
    parser.add_argument('--seq_len',default=60)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch',default=2000)
    parser.add_argument('--lr',default=2e-2)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--model_name', default='CNN', choices=['BattNN', 'LSTM', 'CNN'])
    parser.add_argument('--save_model', default='Lab', choices=[None, 'Lab'])



    args = parser.parse_args()
    return args

from functions import train,test,save_to_txt
from dataloader.load_Lab_data import LabDataSperate,LabData,LabDataFailure

def main():
    args = get_args()
    data = LabData(n=args.batch_size, length=args.seq_len)

    #data = LabDataSperate(n=args.batch_size, length=args.seq_len, test_id=2)
    train_x, train_y = data.load_train_data()
    train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
    error = test(args, data_iter=data.yield_test_data, model_name=args.model_name, plot=5)



def total_main():
    args = get_args()
    for model in ['BattNN', 'LSTM', 'CNN']:
        setattr(args,'model_name',model)
        for id in range(1,9):
            for e in range(5):
                try:
                    data = LabDataSperate(n=args.batch_size,length=args.seq_len,test_id=id)
                    train_x, train_y = data.load_train_data()
                    train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
                    error = test(args, data_iter=data.yield_test_data, model_name=args.model_name,plot=0)
                    [MAE, MAPE, MSE, RMSE] = error
                    info = f'model:{model}. battery:{id}. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
                    print(info)
                    save_to_txt(f'LABdata results.txt', info)
                except:
                    info = f'model:{model}. battery:{id}. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. Error'
                    print(info)
                    save_to_txt(f'LABdata results.txt', info)





if __name__ == '__main__':
    total_main()