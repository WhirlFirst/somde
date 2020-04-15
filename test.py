import pandas as pd
import numpy as np
import NaiveDE,SpatialDE
from somde import SomNode,util

dataname = '../slideseq_data/Puck_180819_11_'
df = pd.read_csv(dataname+'count.csv',sep=',',index_col=1)
corinfo = pd.read_csv(dataname+'idx.csv',sep=',',index_col=0)
del(df['ENSEMBL'])
print(df.shape)
corinfo["total_count"]=df.sum(0)
# stablize,regress_out is gene by cell . However,  run is cell by gene
dfm = NaiveDE.stabilize(df)
res = NaiveDE.regress_out(corinfo, dfm, 'np.log(total_count)').T
X=corinfo[['x','y']].values.astype(np.float32)
som4 = SomNode(X,20)
ndf,ninfo = som4.mtx(df)
r1 ,numberq =som4.run()
nres = som4.norm()
som4.view()