from math import gamma

import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

dataset_name = "flame"

# Load datasets:
X = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/flame.mat')
data = X.get('X')
labels = X.get('labels')


euclidean_distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/flame_euclidean_distances.mat')
D = euclidean_distances.get('D')

distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/d0_distances sin method/lambda_10000/flame_d0_distances.mat')
d0_distances = distances.get('d0_distances')
DMAX = distances.get('DMAX')
DMAX_avg = distances.get('DMAX_avg')
d_best = distances.get('d_best')

Dmax_temp_value = np.amax(D)

# In matlab: T=D+eye(size(D)).*Dmax_temp_value;
T = D + np.diag(np.full(D.shape[1],1)) * Dmax_temp_value
Dmin_temp_value= np.amin(T)
labels = np.reshape(labels, D.shape[1])

# #########################################################
# Compute DBSCAN
distances_interval = np.linspace(2, 10, 10, dtype=int)
distances_interval = np.round(distances_interval)
# Performance measures
db_classic_homogeneity_score = []
db_d0_homogeneity_score = []

NMI_classic = []
NMI_d0 = []

RAND_index_classic = []
RAND_index_d0 = []

V_measure_classic = []
V_measure_d0 = []


for i in distances_interval:
    # Classic DBscan classic:
    db_classic = SpectralClustering(n_clusters=i, assign_labels='discretize', random_state=0).fit(data)
    db_classic_labels_pred = db_classic.labels_
    # D0 DBscan:

    d0_distances= np.exp(-0.001 * d0_distances ** 2)
    db_d0 = SpectralClustering(n_clusters=i, assign_labels='discretize', random_state=0, affinity='precomputed').fit(d0_distances)
    db_d0_labels_pred = db_d0.labels_

    db_classic_homogeneity_score.append(metrics.homogeneity_score(labels, db_classic_labels_pred))
    db_d0_homogeneity_score.append(metrics.homogeneity_score(labels, db_d0_labels_pred))

    NMI_classic.append(metrics.adjusted_mutual_info_score(labels, db_classic_labels_pred))
    NMI_d0.append(metrics.adjusted_mutual_info_score(labels, db_d0_labels_pred))

    RAND_index_classic.append(metrics.rand_score(labels, db_classic_labels_pred))
    RAND_index_d0.append(metrics.rand_score(labels, db_d0_labels_pred))

    V_measure_classic.append(metrics.v_measure_score(labels, db_classic_labels_pred))
    V_measure_d0.append(metrics.v_measure_score(labels, db_d0_labels_pred))

import matplotlib.pyplot as plt

plt.plot(
        distances_interval,
        db_classic_homogeneity_score,
        "r*", label="classic")

plt.plot(distances_interval,
        db_d0_homogeneity_score,
        "b*", label="d0-method")

plt.legend(loc="upper right")
plt.title("{} - Homogeneity - DBscan".format(dataset_name))
plt.xlabel("epsilon distances")
plt.ylabel("homogeneity score")
plt.show()



plt.plot(
        distances_interval,
        NMI_classic,
        "r--", label="classic")
plt.plot(
        distances_interval,
        NMI_d0,
        "b--", label="d0-method")

        #markeredgecolor="k",
        #markersize=10,

plt.legend(loc="upper right")
plt.title("{} - NMI - DBscan".format(dataset_name))
plt.xlabel("epsilon distances")
plt.ylabel("NMI score")
plt.show()

plt.plot(
        distances_interval,
        RAND_index_classic,
        "r--",label="classic")
plt.plot(
        distances_interval,
        RAND_index_d0,
        "b--", label="d0-method")

        #markeredgecolor="k",
        #markersize=10,




plt.legend(loc="upper right")
plt.title("{} - Rand - DBscan".format(dataset_name))
plt.xlabel("epsilon distances")
plt.ylabel("Rand score")
plt.title("Rand")
plt.show()

plt.plot(
        distances_interval,
        V_measure_classic,
        "r--", label="Classic")
plt.plot(
        distances_interval,
        V_measure_d0,
        "b--", label="d0-method")

        #markeredgecolor="k",
        #markersize=10,

plt.legend(loc="upper right")
plt.title("{} - Vmeasure - DBscan".format(dataset_name))
plt.show()

print("debugger point")

# Experimenting
# plt.plot(
#         distances_interval,
#         db_classic_homogeneity_score,
#         "r*", label="classic")
#
#
# plt.legend(loc="upper right")
# plt.title("{} - Homogeneity - DBscan".format(dataset_name))
# plt.xlabel("epsilon distances")
# plt.ylabel("homogeneity score")
# plt.show()