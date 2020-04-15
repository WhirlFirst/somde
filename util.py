import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def plotgene(X,mtx,draw_list,result,sp=10,lw=0.2,N=5,plotsize=5):
    n = len(draw_list)
    rownum = n//N + 1
    plt.figure(figsize=(N*(plotsize+2),plotsize*rownum))
    cmap = LinearSegmentedColormap.from_list('mycmap', ['blue','white','red'])
    for i in range(n):
        if draw_list[i] in list(mtx.T):
            plt.subplot(rownum,N,i+1)
            plt.scatter(X[:,0], X[:,1], c=mtx.T[draw_list[i]],cmap=cmap,s=sp,linewidths=lw,edgecolors='black')
            plt.colorbar()
            if hasattr(result, 'g'):
                plt.title(draw_list[i]+' qval:'+str(round(result[result.g==draw_list[i]].qval.values[0],2)))
            else:
                plt.title(draw_list[i])
        else:
            print('not contain '+str(draw_list[i]))
        
def draw_agree(intersection,r1,r2,verbose=False,N=5):
    r1br2=[]
    r2br1=[]
    all_g=[]
    m1=0
    m2=0
    for i in intersection:
        x1 = r1.index(i)
        x2 = r2.index(i)
        if (abs(x1-x2)>100)&verbose:
            continue
        plt.scatter(x1,x2)
        m1 = max(m1,x1)
        m2 = max(m2,x2)
        if (x1-x2)>N:
            r2br1.append(i)
        elif (x2-x1)>N:
            r1br2.append(i)
        else:
            all_g.append(i)
        plt.annotate("(%s,%s) " %(x1,x2)+str(i), xy=(x1,x2), xytext=(-20, 10), textcoords='offset points')
    plt.plot([0,m2],[N,m2+N],linestyle='-.',color='r')
    plt.plot([N,m2+N],[0,m2],linestyle='-.',color='r')
    plt.plot([0,m2],[10,m2+10],linestyle='-.',color='b')
    plt.plot([10,m2+10],[0,m2],linestyle='-.',color='b')
    plt.xlabel('original')
    plt.xlim(0, m1+10)
    plt.ylim(0, m2+10)
    plt.ylabel('SOM')
    plt.title('Rank 50'+' left_top:'+str(len(r1br2))+' right_down:'+str(len(r2br1))+' all:'+str(len(intersection)))
    return r1br2,r2br1,all_g
    
def draw_agree_log(intersection,r1,r2,label,verbose=False,N=5,al=1000):
    r1br2=[]
    r2br1=[]
    all_g=[]
    m1=0
    m2=0
    x_list=[]
    y_list=[]
    diff=[]
    plt.yscale('log')
    plt.xscale('log')
    plt.axis([1, al, 1, al])
    for i in intersection:
        x1 = r1.index(i)+1
        x2 = r2.index(i)+1
        x_list.append(x1)
        y_list.append(x2)
        diff.append(abs(x1-x2))
        m1 = max(m1,x1)
        m2 = max(m2,x2)
        if (x1-x2)>N:
            r2br1.append(i)
        elif (x2-x1)>N:
            r1br2.append(i)
        else:
            all_g.append(i)
            if x1<10 and x2<10:
                plt.annotate("(%s,%s) " %(x1,x2)+str(i), xy=(x1,x2), xytext=(-20, 10), textcoords='offset points')
            
    plt.scatter(x_list,y_list,c=diff,alpha=0.5,vmin=0,vmax=400)
    print(min(diff),max(diff))

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.colorbar()
    plt.title(label[0]+' VS '+label[1]+' all:'+str(len(intersection)))
    return r1br2,r2br1,all_g