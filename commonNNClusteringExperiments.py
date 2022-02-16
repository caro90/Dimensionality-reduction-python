import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import CommonNNClustering

datasets_dict = load_datasets()

# #########################################################
# Compute DBSCAN
distances_interval = np.linspace(Dmin_temp_value, Dmax_temp_value, 20)
if distances_interval[0] == 0:
    distances_interval = np.delete(distances_interval, 0)

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
    db_classic = CommonNNClustering(eps=i).fit(data)
    db_classic_labels_pred = db_classic.labels_
    # D0 DBscan:
    db_d0 = CommonNNClustering(eps=i, metric="precomputed").fit(d0_distances)
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
        "r--", label="classic")

plt.plot(distances_interval,
        db_d0_homogeneity_score,
        "b--", label="d0-method")

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