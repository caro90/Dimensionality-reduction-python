from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import load_datasets
from sklearn.metrics import f1_score
import numpy as np
from sklearn.manifold import Isomap
from pandas import read_csv


dataset_name = "coil"
datasets_dict = load_datasets()
df = read_csv('coil.csv', delimiter=",", header=None)
data_coil_isomap = df.values

method_name = "DBSCAN"

db_classic_homogeneity_score = []
db_d0_homogeneity_score = []
db_isomap_homogeneity_score = []

NMI_classic = []
NMI_d0 = []
NMI_isomap = []

RAND_index_classic = []
RAND_index_d0 = []
RAND_index_isomap = []

V_measure_classic = []
V_measure_d0 = []
V_measure_isomap = []

f1_classic = []
f1_d0 = []
f1_isomap = []


min_pts = 15

isomap_classic = Isomap(n_components=2).fit_transform(datasets_dict["data"])

for i in datasets_dict["distances_interval"]:

    # db_classic: the result of Dbscan given the self.dist_matrix_ from isomap.py (before dimensionality reduction)
    db_classic = DBSCAN(eps=i, min_samples=min_pts, metric="precomputed").fit(data_coil_isomap)
    db_classic_labels_pred = db_classic.labels_

    # db_d0: contains the result of Dbscan given our precomputed d0_distances
    db_d0 = DBSCAN(eps=i, min_samples=min_pts, metric="precomputed").fit(datasets_dict["d0_distances"])
    db_d0_labels_pred = db_d0.labels_

    # db_isomap: contains the result of DBscan on the output of isomap function
    db_isomap = DBSCAN(eps=i, min_samples=min_pts).fit(isomap_classic)
    db_isomap_labels_pred = db_isomap.labels_

    db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
    db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))
    db_isomap_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_isomap_labels_pred))

    NMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
    NMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))
    NMI_isomap.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_isomap_labels_pred))

    RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
    RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))
    RAND_index_isomap.append(metrics.rand_score(datasets_dict["labels"], db_isomap_labels_pred))

    V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
    V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))
    V_measure_isomap.append(metrics.v_measure_score(datasets_dict["labels"], db_isomap_labels_pred))

    f1_classic.append(f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted'))
    f1_d0.append(f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted'))
    f1_isomap.append(f1_score(datasets_dict["labels"], db_isomap_labels_pred, average='weighted'))

# Plotting
# *******************************************************
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(
    datasets_dict["distances_interval"],
    db_classic_homogeneity_score,
    "r--", label="classic")

ax[0, 0].plot(datasets_dict["distances_interval"],
              db_d0_homogeneity_score,
              "b--", label="d0-method")

ax[0, 0].plot(datasets_dict["distances_interval"],
              db_isomap_homogeneity_score,
              "g--", label="isomap-method")

t = ['d0']
a = [datasets_dict["d_best"].max()]
temp = datasets_dict["distances_interval"].tolist()
temp.append(datasets_dict["d_best"].max())
temp.sort()

for i in range(0, len(temp)):
    if temp[i] == datasets_dict["d_best"].max():
        # temp[i] = 'd0'
        counter = i

    # temp.pop(counter+1)
# t.pop(counter+1)

temp = [np.round(x, 1) for x in temp]

# ax[0, 0].get_xticklabels()[counter].set_color("red")
ax[0, 0].set_xticks(ticks=temp, labels=temp)

ax[0, 0].legend(loc="upper right")
ax[0, 0].set_title("{} - Homogeneity - {}".format(datasets_dict["dataset_name"], method_name))
# ax[0, 0].set_xlabel("epsilon distances")
ax[0, 0].set_ylabel("homogeneity score")

# *******************************************************

ax[0, 1].plot(
    datasets_dict["distances_interval"],
    NMI_classic,
    "r--", label="classic")
ax[0, 1].plot(
    datasets_dict["distances_interval"],
    NMI_d0,
    "b--", label="d0-method")

ax[0, 1].plot(
    datasets_dict["distances_interval"],
    NMI_isomap,
    "g--", label="isomap-method")

# ax[0, 1].get_xticklabels()[counter].set_color("red")
ax[0, 1].set_xticks(ticks=temp, labels=temp)

ax[0, 1].legend(loc="upper right")
ax[0, 1].set_title("{} - NMI - {}".format(datasets_dict["dataset_name"], method_name))
# ax[0, 1].set_xlabel("epsilon distances")
ax[0, 1].set_ylabel("NMI score")

# *******************************************************


ax[1, 0].plot(
    datasets_dict["distances_interval"],
    RAND_index_classic,
    "r--", label="classic")
ax[1, 0].plot(
    datasets_dict["distances_interval"],
    RAND_index_d0,
    "b--", label="d0-method")

ax[1, 0].plot(
    datasets_dict["distances_interval"],
    RAND_index_isomap,
    "g--", label="isomap-method")

# ax[1, 0].get_xticklabels()[counter].set_color("red")
ax[1, 0].set_xticks(ticks=temp, labels=temp)

ax[1, 0].legend(loc="upper right")
ax[1, 0].set_title("{} - Rand - {}".format(datasets_dict["dataset_name"], method_name))
ax[1, 0].set_xlabel("epsilon distances")
ax[1, 0].set_ylabel("Rand score")
# ax[0, 0].show()

# *******************************************************
# fig, ax = plt.subplots()
ax[1, 1].plot(
    datasets_dict["distances_interval"],
    V_measure_classic,
    "r--", label="Classic")
ax[1, 1].plot(
    datasets_dict["distances_interval"],
    V_measure_d0,
    "b--", label="d0-method")

ax[1, 1].plot(
    datasets_dict["distances_interval"],
    V_measure_isomap,
    "g--", label="isomap-method")

# ax[1, 1].get_xticklabels()[counter].set_color("red")
ax[1, 1].set_xticks(ticks=temp, labels=temp)

ax[1, 1].legend(loc="upper right")
ax[1, 1].set_title("{} - Vmeasure - {}".format(datasets_dict["dataset_name"], method_name))
ax[1, 1].set_ylabel("Vmeasure score")
ax[1, 1].set_xlabel("epsilon distances")
plt.show()

plt.plot(
    datasets_dict["distances_interval"],
    f1_classic,
    "r--", label="classic")
plt.plot(
    datasets_dict["distances_interval"],
    f1_d0,
    "b--", label="d0-method")

plt.plot(
    datasets_dict["distances_interval"],
    f1_isomap,
    "g--", label="isomap-method")

# plt.xticklabels()[counter].set_color("red")
# plt.xticks(ticks=temp, labels=temp)

plt.legend(loc="upper right")
plt.title("{} - NMI - {}".format(datasets_dict["dataset_name"], method_name))

plt.ylabel("f1 score")

plt.show()