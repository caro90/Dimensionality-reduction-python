import numpy as np
import scipy.io
import sklearn.manifold
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.manifold import TSNE, Isomap
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset_name = "flame"

# Load datasets:
X = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/Umist.mat')
data = X.get('X')
labels = X.get('label')


euclidean_distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/Umist_euclidean_distances.mat')
D = euclidean_distances.get('D')

distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/d0_distances sin method/Umist_d0_distances.mat')
d0_distances = distances.get('d0_distances')
DMAX = distances.get('DMAX')
DMAX_avg = distances.get('DMAX_avg')
d_best = distances.get('d_best')

Dmax_temp_value = np.amax(D)

# In matlab: T=D+eye(size(D)).*Dmax_temp_value;
T = D + np.diag(np.full(D.shape[1], 1)) * Dmax_temp_value
Dmin_temp_value = np.amin(T)
labels = np.reshape(labels, D.shape[1])

# #########################################################
# Compute DBSCAN
distances_interval = np.linspace(Dmin_temp_value, Dmax_temp_value )
if distances_interval[0] == 0:
    distances_interval = np.delete(distances_interval,0)
# Performance measures
db_classic_homogeneity_score = []
db_d0_homogeneity_score = []

NMI_classic = []
NMI_d0 = []

RAND_index_classic = []
RAND_index_d0 = []

V_measure_classic = []
V_measure_d0 = []

#test = X['X']

# Classic TSNE classic:
isomap_classic = Isomap(n_components=2).fit_transform(data)

# d0-TSNE:
isomap_d0 = TSNE(n_components=2, metric='precomputed').fit_transform(d0_distances)

# Evaluating TSNE results using DBscan:

db_classic = DBSCAN(eps=5, min_samples=15).fit(isomap_classic)
db_classic_labels_pred = db_classic.labels_

db_d0 = DBSCAN(eps=5, min_samples=15).fit(isomap_d0)
db_d0_labels_pred = db_d0.labels_

db_classic_homogeneity_score.append(metrics.homogeneity_score(labels, db_classic_labels_pred))
db_d0_homogeneity_score.append(metrics.homogeneity_score(labels, db_d0_labels_pred))

NMI_classic.append(metrics.adjusted_mutual_info_score(labels, db_classic_labels_pred))
NMI_d0.append(metrics.adjusted_mutual_info_score(labels, db_d0_labels_pred))

RAND_index_classic.append(metrics.rand_score(labels, db_classic_labels_pred))
RAND_index_d0.append(metrics.rand_score(labels, db_d0_labels_pred))

V_measure_classic.append(metrics.v_measure_score(labels, db_classic_labels_pred))
V_measure_d0.append(metrics.v_measure_score(labels, db_d0_labels_pred))



plt.plot(
        isomap_classic[:, 0],
        isomap_classic[:, 1],
        "r*", label="classic")

plt.plot(isomap_d0[:, 0],
         isomap_d0[:, 1],
        "b*", label="d0-method")

plt.legend(loc="upper right")
plt.title("{} - Homogeneity - DBscan".format(dataset_name))
plt.xlabel("epsilon distances")
plt.ylabel("homogeneity score")
plt.show()



