from sklearn.cluster import DBSCAN
from sklearn.manifold import Isomap
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import load_datasets
from mpl_toolkits import mplot3d

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

# Classic TSNE classic:
isomap_classic = Isomap(n_components=2).fit_transform(datasets_dict["data"])

# d0-TSNE:
isomap_d0 = Isomap(n_components=2, metric='precomputed').fit_transform(datasets_dict["d0_distances"])

# Evaluating Isomap results using DBscan:
min_pts = 10
epsilon = 3
db_classic = DBSCAN(eps=epsilon, min_samples=min_pts).fit(isomap_classic)
db_classic_labels_pred = db_classic.labels_

db_d0 = DBSCAN(eps=epsilon, min_samples=min_pts).fit(isomap_d0)
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


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(
        isomap_classic[:, 0],
        isomap_classic[:, 1],
        #isomap_classic[:, 2],
        "r*", label="classic")

ax.plot3D(isomap_d0[:, 0],
         isomap_d0[:, 1],
         #isomap_d0[:, 2],
        "b*", label="d0-method")

plt.legend(loc="upper right")
plt.title("{} - Isomap".format(datasets_dict["dataset_name"]))
plt.show()

plt.scatter(isomap_classic[:, 0],
         isomap_classic[:, 1], c=datasets_dict["labels"],
        label="classic-method")

plt.legend(loc="upper right")
plt.title("{} - Isomap-classic".format(datasets_dict["dataset_name"]))
plt.show()


plt.scatter(isomap_d0[:, 0],
         isomap_d0[:, 1], c=datasets_dict["labels"],
        label="d0-method")

plt.legend(loc="upper right")
plt.title("{} - d0".format(datasets_dict["dataset_name"]))
plt.show()