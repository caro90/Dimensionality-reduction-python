import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# Load datasets:
X = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/coil.mat')
data = X.get('X')
labels = X.get('label')

euclidean_distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/coil_euclidean_distances.mat')
D = euclidean_distances.get('D')

distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/d0_distances sin method/coil_d0_distances.mat')
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
distances_interval = np.linspace(Dmin_temp_value, Dmax_temp_value, 20)

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
    db_classic = DBSCAN(eps=i, min_samples=15).fit(data)
    db_classic_labels_pred = db_classic.labels_
    # D0 DBscan:
    db_d0 = DBSCAN(eps=i, min_samples=15, metric="precomputed").fit(d0_distances)
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
        "r--",
        distances_interval,
        db_d0_homogeneity_score,
        "b--",

        distances_interval,
        NMI_classic,
        "g--",

        distances_interval,
        NMI_d0,
        "m--",

        markeredgecolor="k",
        markersize=10,

    )
plt.title("DBscan")
plt.show()

plt.plot(
        distances_interval,
        RAND_index_classic,
        "r--",
        distances_interval,
        RAND_index_d0,
        "b--",

        markeredgecolor="k",
        markersize=10,

    )
plt.title("Rand")
plt.show()

plt.plot(
        distances_interval,
        V_measure_classic,
        "r--",
        distances_interval,
        V_measure_d0,
        "b--",

        markeredgecolor="k",
        markersize=10,

    )
plt.title("Vmeasure")
plt.show()

print("debugger point")