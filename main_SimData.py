from functions import train,test,save_to_txt
from dataloader.load_sim_data import load_train_data,yield_test_data

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Battery Net for Sim Data')
    parser.add_argument('--V0',default=4.183)
    parser.add_argument('--x0',default=[7856.3254,0,0])
    parser.add_argument('--dt',default=1.0)
    parser.add_argument('--VEOD',default=3.0)

    parser.add_argument('--Rp',default=10000)
    parser.add_argument('--Rs',default=0.1)
    parser.add_argument('--Csp',default=10)
    parser.add_argument('--Cs',default=400)

    parser.add_argument('--batch_size',default=30)
    parser.add_argument('--seq_len',default=60)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch',default=1000)
    parser.add_argument('--lr',default=2e-2)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--model_name',default='BattNN',choices=['BattNN','LSTM','CNN'])
    parser.add_argument('--save_model',default='Sim',choices=[None,'Sim'])


    args = parser.parse_args()
    return args

def main():
    args = get_args()
    train_x,train_y = load_train_data(n=args.batch_size,length=args.seq_len)
    loss = train(args=args,train_x=train_x,train_y=train_y,model_name=args.model_name)
    error = test(args,data_iter=yield_test_data,model_name=args.model_name,plot=0)

def total_main():
    args = get_args()
    for model in ['CNN']:
        setattr(args,'model_name',model)
        for e in range(5):
            try:
                train_x,train_y = load_train_data(n=args.batch_size,length=args.seq_len)
                train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
                error = test(args,data_iter=yield_test_data,model_name=args.model_name,plot=0)
                [MAE, MAPE, MSE, RMSE] = error
                info = f'model:{model}. battery:None. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
                print(info)
                save_to_txt(f'Simdata results CNN.txt', info)

            except:
                info = f'model:{model}. battery:None. batch size:{args.batch_size}. seq len:{args.seq_len}. experiment={e + 1}. Error'
                print(info)
                save_to_txt(f'Simdata results CNN.txt', info)



def diffenent_seq_len():
    args = get_args()
    setattr(args, 'Rp', 10000)
    setattr(args, 'Rs', 0.1)
    setattr(args, 'Csp', 10)
    setattr(args, 'Cs', 400)

    for model in ['LSTM','LSTM']:
        setattr(args, 'model_name',model)
        for l in [20,30,40,50,60,70,80,90,100]:
            setattr(args, 'seq_len', l)
            for e in range(5):
                try:
                    train_x, train_y = load_train_data(n=args.batch_size, length=args.seq_len)
                    train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
                    error = test(args, data_iter=yield_test_data, model_name=args.model_name,plot=0)
                    [MAE, MAPE, MSE, RMSE] = error
                    info = f'model:{model}. batch size:{args.batch_size}. seq len: {l}. experiment={e + 1}. MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
                    print(info)
                    save_to_txt(f'Simdata_{model}_seq_len.txt', info)
                except:
                    info = f'model:{model}. batch size:{args.batch_size}. seq len: {l}. experiment={e + 1}. Error'
                    print(info)
                    save_to_txt(f'Simdata_{model}_seq_len.txt', info)

def different_batch_size():
    args = get_args()
    for model in ['BattNN','LSTM']:
        setattr(args, 'model_name', model)
        for bs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            setattr(args, 'batch_size', bs)
            for e in range(5):
                try:
                    train_x, train_y = load_train_data(n=args.batch_size, length=args.seq_len)
                    train(args=args, train_x=train_x, train_y=train_y, model_name=args.model_name)
                    error = test(args, data_iter=yield_test_data, model_name=args.model_name, plot=0)
                    [MAE, MAPE, MSE, RMSE] = error
                    info = f'model:{model}. batch size:{bs}. seq len:{args.seq_len}. experiment={e + 1}. MAE={MAE}. MAPE={MAPE}. MSE={MSE}. RMSE={RMSE}'
                    print(info)
                    save_to_txt(f'Simdata_{model}_batch_size.txt', info)
                except:
                    info = f'model:{model}. batch size:{bs}. seq len:{args.seq_len}. experiment={e + 1}. Error'
                    print(info)
                    save_to_txt(f'Simdata_{model}_batch_size.txt', info)



if __name__ == '__main__':
    main()