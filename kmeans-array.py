import pandas as pd
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sb
%matplotlib inline
df=pd.read_csv('class01.csv')

lista1 = np.array([ 1 for x in range(101)])
listaneg_1 = np.array([ -1 for x in range(101)])
lista_alter = np.array([((-1) ** x) for x in range(101)])

centroid= [lista1, listaneg_1 , lista_alter]
init_center=np.array(centroid, np.int32)
kmeans = KMeans(n_clusters=3, init=init_center, max_iter=10, random_state=0).fit(df)
