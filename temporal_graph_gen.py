import numpy as np
import pandas as pd
import time
import argparse

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=3,normal=False):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="jam", help="Dataset path.")
parser.add_argument("--order", type=int, default=1, help="DTW order.")
parser.add_argument("--lag", type=int, default=12, help="Fast DTW search lag.")
parser.add_argument("--period", type=int, default=288, help="Time series perios.")
parser.add_argument("--sparsity", type=float, default=0.0075, help="sparsity of spatial graph")
parser.add_argument("--n_his", type=int, default=12, help="number of history time slots fed into the model")

args = parser.parse_args()

df = pd.read_csv(args.dataset+'.csv')
#shape of df is (time_slots * nodes)
df = df.drop(columns=['date'])
time_slots, nodes = df.shape

print(time_slots, nodes)
num_train = int(time_slots * 0.6)
num_dtw = int(num_train/args.n_his)*args.n_his
df = df.iloc[:num_dtw,:]
data = df.values.reshape(-1, args.n_his, nodes)
d=np.zeros([nodes,nodes])

print(df.head(10))
for i in range(nodes):
    t1=time.time()
    for j in range(i+1,nodes):
        d[i,j]=compute_dtw(data[:,:,i],data[:,:,j],order=args.order,Ts=args.lag)
    t2=time.time()
    print('Line',i,'finished in',t2-t1,'seconds.')

dtw=d+d.T

np.save(args.dataset+"_dtw.npy",dtw)
print("The calculation of time series is done!")

adj = np.load(args.dataset+"_dtw.npy")
adj = adj+ adj.T

w_adj = np.zeros([nodes,nodes])

adj_percent = args.sparsity

top = int(nodes * adj_percent)
for i in range(adj.shape[0]):
    a = adj[i,:].argsort()[0:top]
    for j in range(top):
        w_adj[i, a[j]] = 1

for i in range(nodes):
    for j in range(nodes):
        if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
            w_adj[i][j] = 1
        if(i == j):
            w_adj[i][j] = 1

print("Total route number: ", nodes)
print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(nodes*nodes))

pd.DataFrame(w_adj).to_csv(args.dataset+"_dtw.csv", index = False, header=None)

print("The weighted matrix of temporal graph is generated!")
