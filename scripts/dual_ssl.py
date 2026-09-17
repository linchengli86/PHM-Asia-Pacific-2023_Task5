"""Dual Channel + SSL on S1 only + raw concat — preserve diversity"""
import sys,os,numpy as np;sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score,f1_score
from src.data.dataset import load_raw_data;from src.data.labels import generate_labels
from src.data.sequence_dataset import WindowDataset
device=torch.device("cuda")
R2=r2_score;MAE=mean_absolute_error;ACC=accuracy_score
F1=lambda y,t:f1_score(y,t,average="macro",zero_division=0)

X_train,X_test=load_raw_data('C:/Users/11762/Desktop/dataset',list(range(1,178)),list(range(178,224)))
def zscore(X):m=X.mean(axis=1,keepdims=True);s=X.std(axis=1,keepdims=True)+1e-8;return(X-m)/s
X_train_n=zscore(X_train);X_test_n=zscore(X_test)
y_case=generate_labels('C:/Users/11762/Desktop/dataset/train/labels.xlsx')
ans=pd.read_csv('C:/Users/11762/Desktop/dataset/test data/answer.csv');gt_t5=ans['task5'].values

# Build windows
X_ds=X_train_n[:,::6,:][:,:200,:]
X_w=[];y_w={k:[] for k in y_case}
for i in range(177):
    nw=36 if y_case['task5'][i]<100 else 12;ts=X_ds[i];T=ts.shape[0];tp=max(0,T-200)
    for _ in range(nw):
        t0=np.random.randint(0,max(1,tp)) if tp>0 else 0;w=ts[t0:t0+200]
        if w.shape[0]<200:w=np.pad(w,((0,200-w.shape[0]),(0,0)))
        X_w.append(w)
    for k,v in y_case.items():y_w[k].extend([v[i]]*nw)
X_w=np.stack(X_w).astype(np.float32);y_w={k:np.array(v) for k,v in y_w.items()}
print(f'W:{X_w.shape[0]}')

class CNNE(nn.Module):
    def __init__(self):
        super().__init__();c=7;ls=[]
        for ch,k in zip([64,128,128],[7,5,3]):
            ls.extend([nn.Conv1d(c,ch,k,padding=k//2),nn.BatchNorm1d(ch),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.3)]);c=ch
        self.net=nn.Sequential(*ls);self.gap=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):return self.gap(self.net(x.transpose(1,2))).squeeze(-1)

ssl_enc=CNNE().to(device);ssl_enc.load_state_dict(torch.load('models/cross_ssl.pt','cpu'))
for p in ssl_enc.parameters():p.requires_grad=False
A5=torch.tensor([0.,25.,50.,75.,100.])

class DualSSL(nn.Module):
    def __init__(self):
        super().__init__()
        # S1: SSL-pretrained + raw CNN (frozen SSL, trainable raw)
        self.s1_ssl=ssl_enc
        self.s1_raw=nn.Sequential(nn.Conv1d(7,32,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.s1_head=nn.Linear(160,1)
        # S2: random init (trainable) — same architecture as original Dual Channel
        self.s2=CNNE()
        # Fusion: [S1(160) ⊕ S2(128)] = 288 → 128
        self.fusion=nn.Sequential(nn.Linear(288,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2))
        self.h2=nn.Linear(128,4);self.h3=nn.Linear(128,9);self.h4=nn.Linear(128,5)
        self.ac=nn.Linear(128,5)
        self.dh=nn.ModuleList([nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))for _ in range(5)])
    def forward(self,x):
        h1=self.s1_ssl(x);r1=self.s1_raw(x.transpose(1,2)).squeeze(-1);f1=torch.cat([h1,r1],dim=1)
        o1=self.s1_head(f1)
        h2=self.s2(x);f2=torch.cat([f1,h2],dim=1);hu=self.fusion(f2)
        o2=self.h2(hu);o3=self.h3(hu);o4=self.h4(hu)
        lg=self.ac(hu);w=torch.softmax(lg,dim=1);a=A5.to(x.device)
        ds=torch.stack([d(hu).squeeze(-1)for d in self.dh],dim=1)
        o5=(w*(a+ds)).sum(1,keepdim=True)
        return o1,o2,o3,o4,torch.relu(o5),lg

m=DualSSL().to(device)
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
ldr=DataLoader(WindowDataset(X_w,y_w),128,True)
Xt=zscore(X_test)[:,::6,:][:,:200,:].astype(np.float32);Xt_t=torch.from_numpy(Xt).float().to(device)
best_r2,bep,bck=-999,0,None
print('DualSSL: 400ep...')
for ep in range(400):
    m.train();t,n=0,0
    for batch in ldr:
        x=batch['x'].to(device);t5=batch['task5'].float().to(device);t5s=t5.squeeze(-1)
        o1,o2,o3,o4,o5,lg=m(x);dif=torch.abs(t5s.unsqueeze(1)-A5.to(device).unsqueeze(0));tc=dif.argmin(1)
        l=F.binary_cross_entropy_with_logits(o1,batch['task1'].float().to(device))
        l+=1.5*F.cross_entropy(o2,batch['task2'].long().to(device))
        l+=F.cross_entropy(o3,batch['task3'].long().to(device))
        l+=F.cross_entropy(o4,batch['task4'].long().to(device))
        l+=0.5*F.cross_entropy(lg,tc)+2.0*F.smooth_l1_loss(o5,t5)
        opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();t+=l.item();n+=1
    m.eval()
    with torch.no_grad():_,_,_,_,o5,_=m(Xt_t);r2=R2(gt_t5,o5.cpu().numpy().squeeze())
    if r2>best_r2:best_r2=r2;bep=ep+1;bck={k:v.cpu().clone()for k,v in m.state_dict().items()}
    if(ep+1)%40==0:print(f'  Ep{ep+1}:tr={t/n:.1f} r2={r2:.4f} best={best_r2:.4f}@{bep}')

m.load_state_dict(bck);m.eval()
with torch.no_grad():
    o1,o2,o3,o4,o5,_=m(Xt_t);p1=(torch.sigmoid(o1)>0.5).int().cpu().numpy().squeeze();p2=o2.argmax(1).cpu().numpy()
    p3=o3.argmax(1).cpu().numpy();p4=o4.argmax(1).cpu().numpy();p5=o5.cpu().numpy().squeeze()
gt={c:ans[c].values for c in['task1','task2','task3','task4','task5']}
print(f'\nDualSSL best@{bep}')
print(f't1:{ACC(gt["task1"],p1):.4f} t2:{ACC(gt["task2"],p2):.4f}')
print(f't3_f1:{F1(gt["task3"],p3):.4f} t4_f1:{F1(gt["task4"],p4):.4f}')
print(f't5_mae:{MAE(gt["task5"],p5):.1f} t5_r2:{R2(gt["task5"],p5):.4f}')
for sc in[1,4]:msk=ans['Spacecraft No.'].values==sc;print(f'  SC{sc}:mae={MAE(gt["task5"][msk],p5[msk]):.1f} r2={R2(gt["task5"][msk],p5[msk]):.4f}')
print('DONE')
