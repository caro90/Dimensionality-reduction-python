import numpy as np
import scipy.io
import sklearn.manifold
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.manifold import TSNE, Isomap
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from loadDatasets import load_datasets

datasets_dict = load_datasets()

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
isomap_classic = Isomap(n_components=2).fit_transform(datasets_dict["data"])

# d0-TSNE:
isomap_d0 = TSNE(n_components=2, metric='precomputed').fit_transform(datasets_dict["d0_distances"])

# Evaluating TSNE results using DBscan:

db_classic = DBSCAN(eps=5, min_samples=15).fit(isomap_classic)
db_classic_labels_pred = db_classic.labels_

db_d0 = DBSCAN(eps=5, min_samples=15).fit(isomap_d0)
db_d0_labels_pred = db_d0.labels_

db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))

print("Classic homogeneity:", db_classic_homogeneity_score)
print("d0 homogeneity:", db_d0_homogeneity_score)

NMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
NMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))

print("NMI :", NMI_classic)
print("NMI d0 :", NMI_d0)

RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))

print("RAND_index_classic:", RAND_index_classic)
print("d0 RAND_index_classic:", RAND_index_d0)

V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))

print("V_measure_classic:", V_measure_classic)
print("d0 V_measure :", V_measure_d0)



plt.plot(
        isomap_classic[:, 0],
        isomap_classic[:, 1],
        "r*", label="classic")

plt.plot(isomap_d0[:, 0],
         isomap_d0[:, 1],
        "b*", label="d0-method")

plt.legend(loc="upper right")
plt.title("{} - Isomap".format(datasets_dict["dataset_name"]))
plt.xlabel("epsilon distances")
plt.ylabel("homogeneity score")
plt.show()



