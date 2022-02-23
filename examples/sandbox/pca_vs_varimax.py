import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets

try:
	from clif import fingerprints
except:
	import sys
	sys.path.append("../clif/")
	from eof import fingerprints


# load data set
X = datasets.load_digits().data
X0 = X - np.mean(X,axis=0) # center
n_samples, n_dim = X.shape

# obtain fingerprints
n_components = 8
fp = fingerprints(n_eofs=n_components,varimax=True)
fp.fit(X)

eofs_pca = fp.eofs_
eofs_varimax = fp.eofs_varimax_

pca_explained_variance = fp.explained_variance_
varimax_explained_variance = fp.explained_variance_varimax_

plt.plot(pca_explained_variance)
plt.plot(varimax_explained_variance)
plt.show()
