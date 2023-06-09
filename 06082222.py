import numpy as np
import pandas as pd
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,Dataset
from tqdm import trange


def my_collate(batch):
    inputs,target, haha, y_lag = zip(*batch)
    inputs = [torch.from_numpy(bat) for bat in inputs]
    target = [torch.from_numpy(bat) for bat in target]
    haha = [torch.from_numpy(bat) for bat in haha]
    y_lag = [torch.from_numpy(bat) for bat in y_lag]
    return inputs, target,haha, y_lag


class MyDataset(Dataset):
    def __init__(self,a,b,c,d):
        self.x = a
        self.y = b
        self.z = c
        self.w = d
        self.len = len(self.x)
 
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.z[index], self.w[index]
 
    def __len__(self):
        return self.len

def make_data(data,factor_mode,batch_size, shuffle):
    data_=data.groupby('date')
    x_beta=[]
    y=[]
    x_factor=[]
    y_lag=[]
    for tmp in data_:
        n_tickers=len(tmp[1]['permno'])
        X_beta = tmp[1][characteristics].values.reshape(-1,n_tickers, n_characteristics)
        x_beta.append(X_beta)
        X_factor_ret=tmp[1]['ret-rf'].values.reshape(-1,n_tickers,1)
        y.append(X_factor_ret)

        lag_return=tmp[1]['ret-rf-new'].values.reshape(-1,n_tickers,1)
        y_lag.append(lag_return)

        if factor_mode=='folios':
            Z_t_minus_1 = X_beta
            X_factor=np.linalg.pinv(Z_t_minus_1.transpose(0,2,1) @ Z_t_minus_1) @ Z_t_minus_1.transpose(0,2,1) @ X_factor_ret
            # print(X_factor.shape)
            x_factor.append(X_factor.reshape(-1,1,n_characteristics))
        elif factor_mode=='return':

            x_factor.append(X_factor_ret.reshape(-1,1,n_tickers))
        else:
            raise ValueError('factor_mode must be either folios or return')

    print(len(x_beta))
    
    return torch.utils.data.DataLoader(MyDataset(x_beta,x_factor,y,y_lag),collate_fn=my_collate, batch_size=batch_size, shuffle=shuffle,drop_last=False)

class CA(nn.Module):
    def __init__(self, stocks_num,character_num, encoding_dim, CA_level='CA1',factor_mode='return',batch_size=1):
        super(CA, self).__init__()
        
        self.stocks_num=stocks_num
        self.character_num=character_num
        self.encoding_dim=encoding_dim
        self.factor_mode=factor_mode
        self.batch_size=batch_size
        
        self.beta_line_CA0=nn.Sequential(
            nn.Linear(character_num, encoding_dim)
        )
        self.beta_line_CA1=nn.Sequential(
            nn.Linear(character_num, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        self.beta_line_CA2=nn.Sequential(
            nn.Linear(character_num, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Linear(16, encoding_dim)
        )
        self.beta_line_CA3=nn.Sequential(
            nn.Linear(character_num, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),  
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Linear(8, encoding_dim)
        )
        
        self.beta_line_CA_dict={'CA0':self.beta_line_CA0,
                                'CA1':self.beta_line_CA1,
                                'CA2':self.beta_line_CA2,
                                'CA3':self.beta_line_CA3}
        
        
        self.beta_line=self.beta_line_CA_dict[CA_level]
        self.factor_mode=factor_mode

        if self.factor_mode=='return':
            self.factor_line=nn.Sequential(
                nn.Linear(self.stocks_num, self.encoding_dim),
            ).to(device)

        elif self.factor_mode=='folios':
            self.factor_line=nn.Sequential(
                nn.Linear(self.character_num, self.encoding_dim),
            ).to(device)

        else:
            raise ValueError('factor_mode must be either folios or return')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, beta, factor,y):
        ww= []
        los=0
        cnt=0

        for i, j, k in zip(beta, factor, y):
            y1=y[cnt].reshape(-1)
            self.stocks_num=y[cnt].shape[1]
            factor1 = self.factor_line(factor[cnt].reshape(-1)).view(self.encoding_dim,1)
            loss=0
            beta_= self.beta_line(beta[cnt].reshape(self.stocks_num, self.character_num)).reshape(self.stocks_num, self.encoding_dim)
            w = beta_ @ factor1
            w = w.reshape(-1)
            loss+=criterion(y1,w)
            los+=loss*100
            cnt+=1
            ww.append(w)
        return los/cnt,ww
    
def run(numnum,ca,type):
    model=CA(n_tickers,n_characteristics,encoding_dim=numnum,CA_level=ca,factor_mode=type,batch_size=16).to(device)
    loss_train=[]
    loss_val=[]
    acc_train=[]
    acc_val=[]                
    r2_=[]
    r2_test=[]

    lag_r2_=[]
    lag_r2_test=[]

    for epoch in range(EPOCH):
        
        model.train()
        train_loss=0
        train_acc=0
        numm = 0
        
        for i,x in enumerate(train_loader):
            # print(i)
            x_beta,x_factor_ret,y,y_lag=x
            x_beta = [a.to(device).to(torch.float32) for a in x_beta]
            x_factor_ret = [a.to(device).to(torch.float32) for a in x_factor_ret]
            y = [a.to(device).to(torch.float32) for a in y]
            y_lag = [a.to(device).to(torch.float32) for a in y_lag]
            model.optimizer.zero_grad()
            loss,y_pred=model(x_beta,x_factor_ret,y)
            

            # regularization_loss=torch.tensor(0.0).to(device)
            # for param in model.parameters():
            #     regularization_loss+=torch.sum(torch.abs(param))
            # loss+=LASSO_lamb*regularization_loss

            loss.backward()
            model.optimizer.step()
            # y_new = y.reshape(-1)
            train_loss+=loss.item()
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        print('Epoch: ',epoch,' Train Loss: ',loss_train[-1])
        
        val_loss=0
        test_loss=0
        val_acc=0
        model.eval()
        
        r2_tmp=0
        up_tmp = 0
        down_tmp = 0
        cnt = 0
        y_pred_all=np.array([])
        y_true_all=np.array([])
        y_lag_all=np.array([])
        with torch.no_grad():
            for i,x in enumerate(val_loader):
                x_beta,x_factor_ret,y,y_lag=x
                x_beta = [a.to(device).to(torch.float32) for a in x_beta]
                x_factor_ret = [a.to(device).to(torch.float32) for a in x_factor_ret]
                y = [a.to(device).to(torch.float32) for a in y]
                y_lag = [a.to(device).to(torch.float32) for a in y_lag]
                # print(list(y[0]))
                # break
                # print("uejdh")
                loss,y_pred=model(x_beta,x_factor_ret,y)
                # print(y[0].shape)
                # regularization_loss=torch.tensor(0.0).to(device)
                # for param in model.parameters():
                #     regularization_loss+=torch.sum(torch.abs(param))
                # loss+=LASSO_lamb*regularization_loss
                for k in y_pred:
                    y_pred_all = np.append(y_pred_all, k.cpu().detach().numpy())
                for k in y:
                    # b = k.cpu().detach().numpy().reshape(-1)
                    y_true_all = np.append(y_true_all,k.cpu().detach().numpy().reshape(-1))    
                for k in y_lag:
                    # b = k.cpu().detach().numpy().reshape(-1)
                    y_lag_all = np.append(y_lag_all,k.cpu().detach().numpy().reshape(-1))    
                
                val_loss+=loss.item()
                cnt += 1
            loss_val.append(val_loss)
            acc_val.append(val_acc)
            aaa = y_pred_all - y_true_all
            bbb = aaa.T@aaa
            ccc = y_true_all.T@y_true_all
            ddd = bbb/ccc
            r22=1-ddd

            eee = y_pred_all - y_lag_all
            fff = eee.T @ eee
            ggg = y_lag_all.T @ y_lag_all
            hhh = fff/ggg
            r22_lag = 1 - hhh

            r2_.append(r22)
            lag_r2_.append(r22_lag)
            print('Epoch: ',epoch,' Val Loss: ',loss_val[-1],' Val Acc: ',acc_val[-1],'R2:',r22, 'R2_lag:',r22_lag)

            test_acc=0
            test_loss=0
            y_new=0
            y_pred=0
            r2_tmp=0
            y_pred_all=np.array([])
            y_true_all=np.array([]) 
            y_lag_all=np.array([])
            for i,x in enumerate(test_loader):
                x_beta,x_factor_ret,y, y_lag=x
                x_beta = [a.to(device).to(torch.float32) for a in x_beta]
                x_factor_ret = [a.to(device).to(torch.float32) for a in x_factor_ret]
                y = [a.to(device).to(torch.float32) for a in y]
                y_lag = [a.to(device).to(torch.float32) for a in y_lag]
                
                loss,y_pred=model(x_beta,x_factor_ret,y)
                for k in y_pred:
                    y_pred_all = np.append(y_pred_all, k.cpu().detach().numpy())
                for k in y:
                    # b = k.cpu().detach().numpy().reshape(-1)
                    y_true_all = np.append(y_true_all,k.cpu().detach().numpy().reshape(-1))    
                for k in y_lag:
                    # b = k.cpu().detach().numpy().reshape(-1)
                    y_lag_all = np.append(y_lag_all,k.cpu().detach().numpy().reshape(-1))    

            aaa = y_pred_all - y_true_all
            bbb = aaa.T@aaa
            ccc = y_true_all.T@y_true_all
            ddd = bbb/ccc
            r23=1-ddd

            eee = y_pred_all - y_lag_all
            fff = eee.T @ eee
            ggg = y_lag_all.T @ y_lag_all
            hhh = fff/ggg
            r23_lag = 1 - hhh


            # print('Test R2:',r23,'Test R2_lag:',r23_lag)
        # torch.save(model.state_dict(), f'model_{epoch}.pt')

            r2_test.append(r23)
            lag_r2_test.append(r23_lag)

    c = {'val':r2_,'test':r2_test,'val_lag':lag_r2_,'test_lag':lag_r2_test}
    df = pd.DataFrame(c)
    df.to_csv(str(numnum)+str(ca)+str(type)+'China.csv')
    del model
    return


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # warnings.filterwarnings('ignore')
    characteristics = ['Opndt', 'Mopnprc', 'Mclsprc', 'Mnshrtrd', 'Mnvaltrd', 'Msmvosd']

    # df_final=pd.read_csv('./guiyi_.csv')
    df_final=pd.read_feather('China.feather')

    firm_all=df_final['permno'].unique()
    n_characteristics=len(characteristics)

    start_date_train = 199101
    end_date_train = 200912
    start_date_val = 201001
    end_date_val = 201612
    start_date_test = 201701
    end_date_test = 202212

    data_push_train=df_final[(df_final['date'] >= start_date_train) & (df_final['date'] < end_date_train)]
    data_push_val=df_final[(df_final['date'] >= start_date_val) & (df_final['date'] < end_date_val)]
    train_loader = make_data(data_push_train,factor_mode='folios',batch_size=16, shuffle=True)
    val_loader = make_data(data_push_val,factor_mode='folios',batch_size=16, shuffle=False)

    data_push_test=df_final[(df_final['date'] >= start_date_test) & (df_final['date'] < end_date_test)]
    test_loader = make_data(data_push_test,factor_mode='folios',batch_size=16, shuffle=False)

    criterion=nn.MSELoss().to(device)

    EPOCH=300
    factor_mode='folios'
    batch_size=1
    shuffle=False
    n_tickers=len(firm_all)


    # print(model)
    LASSO_lamb=0.1
    early_stopping=False
    PATIENCE=20
    r2=[]

    for i in range (1,5):
        run(numnum=i,ca='CA3',type='folios')
        run(numnum=i,ca='CA2',type='folios')
        run(numnum=i,ca='CA1',type='folios')
        run(numnum=i,ca='CA0',type='folios')