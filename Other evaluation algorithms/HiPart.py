from HiPart.clustering import DePDDP
from sklearn.datasets import make_blobs

# Running algorithms from the HiPart package
# Source and documentation: https://hipart.readthedocs.io/index.html

X, y = make_blobs(n_samples=1500, centers=6, random_state=0)
clustered_class = DePDDP(max_clusters_number=6).fit_predict(X)


