
# SOMDE ![pypi](https://img.shields.io/pypi/v/somde)
Algorithm for finding gene spatial pattern based on Gaussian process accelerated by SOM

## Install

```bash
pip install numpy
pip install somde
```

## Data
Slide-seq data we used can be downloaded from SpatialDB website:
http://www.spatialomics.org/SpatialDB/download.php

## Tutorial


### load data
```python
df = pd.read_csv(dataname+'count.csv',sep=',',index_col=1)
corinfo = pd.read_csv(dataname+'idx.csv',sep=',',index_col=0)
corinfo["total_count"]=df.sum(0)
X=corinfo[['x','y']].values.astype(np.float32)
```
After data loading, we can generate a SOM on the tissue spatial domain.
### build SOM
```python
from somde import SomNode
som = SomNode(X,20)
```
You can use `som.view()` to visualize the distribution of all SOM nodes.

### integrate data sites and expression
```python
ndf,ninfo = som.mtx(df)
```
`mtx`function will generate pesudo gene expression and spatial data site information at reduced resolution.

### normalize data and identify SVgenes
Since we integrated the original count data, we need to normalize gene expression matrix in each `SomNode` object.
```python
nres = som.norm()
result, SVnum =som.run()
```
The identification step is mainly based on the adjusted Gaussian Process, which was first proposed by [SpatialDE](https://github.com/Teichlab/SpatialDE).
Visualization results can be found at https://github.com/WhirlFirst/somde/blob/master/slide_seq0819_11_SOM.ipynb 


