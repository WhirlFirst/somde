import somoclu
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from matplotlib.colors import LinearSegmentedColormap
import NaiveDE,SpatialDE

class SomNode:
    def __init__(self, X, k,homogeneous_codebook=True):
        self.X = X
        self.somn = int(np.sqrt(X.shape[0]//k))
        self.ndf = None
        self.ninfo = None
        self.nres=None
        if homogeneous_codebook:
            xmin,ymin = X.min(0)
            xmax,ymax = X.max(0)
            cobx,coby = np.meshgrid(np.linspace(xmin,xmax,self.somn),np.linspace(ymin,ymax,self.somn))
            self.inicodebook = np.transpose(np.array([cobx.ravel(),coby.ravel()],np.float32),(1,0))
            print('using {0}*{0} SOM nodes for {1} points'.format(self.somn,X.shape[0]))
            self.som = somoclu.Somoclu(self.somn, self.somn,initialcodebook=self.inicodebook.copy())
        else:
            self.som = somoclu.Somoclu(self.somn, self.somn)
        self.som.train(X,epochs=10)
        
    def reTrain(self,ep):
        self.som.train(X,epochs=ep)
        
        
    def viewIniCodebook(self,orisize=3,codesize=20):
        plt.scatter(self.X[:,0], self.X[:,1],s=orisize, label='original')
        plt.scatter(self.inicodebook[:,0], self.inicodebook[:,1],s=codesize, label='initialCodebook')
        plt.legend()
        
        
    def view(self,raw = True,c=False,line=False):
        rr = self.som.codebook
        sizenum = np.ones([rr.shape[0],rr.shape[1]])*30
        rr = np.reshape(rr,[-1,2])
        if raw:
            plt.scatter(self.X[:,0], self.X[:,1],s=3, label='original')
        for i in range(self.X.shape[0]):
            v,u = self.som.bmus[i]
            sizenum[u,v] +=2
            if line:
                plt.plot([self.X[i,0],self.som.codebook[u,v,0]],[self.X[i,1],self.som.codebook[u,v,1]])
        sizenum = np.reshape(sizenum,[-1,])
        if c:
            plt.scatter(rr[:,0],rr[:,1],s=sizenum,label=str(self.somn)+'X'+str(self.somn)+' SOM nodes',c=sizenum,cmap='hot')
            plt.colorbar()
        else:
            plt.scatter(rr[:,0],rr[:,1],s=sizenum,label=str(self.somn)+'X'+str(self.somn)+' SOM nodes',c='r')
        plt.legend()
        
    def mtx(self,df,alpha=0.5):
        bsmc = self.som.bmus
        soml = []
        for i in np.arange(bsmc.shape[0]):
            u,v = bsmc[i]
            soml.append(v*self.somn+u)
        ndf_value = np.zeros([df.shape[0],len(np.unique(np.array(soml)))])
        ninfo = pd.DataFrame(columns=['x','y','total_count'])
        tmp=0
        for i in np.unique(np.array(soml)):
            select_df = df.loc[:,np.array(soml)==i]
            ndf_value[:,tmp]=alpha*select_df.max(1)+(1-alpha)*select_df.mean(1)
            coor = self.som.codebook[i//self.somn,i%self.somn]
            ninfo.loc[tmp,'x'] = round(coor[0],4)
            ninfo.loc[tmp,'y'] = round(coor[1],4)
            tmp+=1
        ndf = pd.DataFrame(ndf_value,index= list(df.T))
        ninfo.total_count = ndf.sum(0)
        self.ndf = ndf
        self.ninfo = ninfo
        return ndf,ninfo
    
    def norm(self):
        if self.ninfo is None:
            print('generate mtx first')
            return
        dfm = NaiveDE.stabilize(self.ndf)
        self.nres = NaiveDE.regress_out(self.ninfo, dfm, 'np.log(total_count)').T
        return self.nres
    
    def run(self):
        if self.nres is None:
            print('norm mtx first')
            self.norm()
        X=self.ninfo[['x','y']].values.astype(float)
        result = SpatialDE.run(X, self.nres)
        result.sort_values('LLR',inplace=True,ascending=False)
        number_q = result[result.qval<0.05].shape[0]
        return result, number_q