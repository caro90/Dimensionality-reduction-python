from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import load_datasets
from sklearn_extra.cluster import KMedoids
from sklearn_extra.cluster import CommonNNClustering
import numpy as np

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

# Choose clustering method:
# - set clustering_method to 1 for DBscan
# - set clustering_method to 2 for OPTICS
# - set clustering_method to 3 for commonNN
# - set clustering_method to 4 for kMedoid
clustering_method = 3

if clustering_method == 1:
    method_name = "DBSCAN"
elif clustering_method == 2:
    method_name = "OPTICS"
elif clustering_method == 3:
    method_name = "CommonNN"


# DBSCAN parameter:
min_pts = 15

for i in datasets_dict["distances_interval"]:

    if clustering_method == 1:
        # Classic DBscan approach:
        db_classic = DBSCAN(eps=i, min_samples=min_pts).fit(datasets_dict["data"])
        db_classic_labels_pred = db_classic.labels_
        # d0 DBscan:
        db_d0 = DBSCAN(eps=i, min_samples=min_pts, metric="precomputed").fit(datasets_dict["d0_distances"])
        db_d0_labels_pred = db_d0.labels_

    elif clustering_method == 2:
        # Classic OPTICS approach
        print("Optics: {}".format(i))
        db_classic = OPTICS(eps=i, min_samples=5).fit(datasets_dict["data"])
        db_classic_labels_pred = db_classic.labels_
        # d0 OPTICS:
        db_d0 = OPTICS(eps=i, min_samples=15, metric="precomputed").fit(datasets_dict["d0_distances"])
        db_d0_labels_pred = db_d0.labels_

    elif clustering_method == 3:
        # Classic CommonNN:
        db_classic = CommonNNClustering(eps=i).fit(datasets_dict["data"])
        db_classic_labels_pred = db_classic.labels_
        # D0 CommonNN:
        db_d0 = CommonNNClustering(eps=i, metric="precomputed").fit(datasets_dict["d0_distances"])
        db_d0_labels_pred = db_d0.labels_

    elif clustering_method == 4:
        # Classic KMedoids:
        db_classic = KMedoids(n_clusters=i).fit(datasets_dict["data"])
        db_classic_labels_pred = db_classic.labels_
        # D0 KMedoids:
        db_d0 = KMedoids(n_clusters=i, metric="precomputed").fit(datasets_dict["d0_distances"])
        db_d0_labels_pred = db_d0.labels_

    db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
    db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))

    NMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
    NMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))

    RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
    RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))

    V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
    V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))

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


t = ['d0']
a = [datasets_dict["d_best"].max()]
temp = datasets_dict["distances_interval"].tolist()
temp.append(datasets_dict["d_best"].max())
temp.sort()
#temp_label = temp
#temp = [int(x) for x in temp]
#t = [str(n) for n in temp]

for i in range(0, len(temp)):
    if temp[i] == datasets_dict["d_best"].max():
        #temp[i] = 'd0'
        counter = i


#temp.pop(counter+1)
#t.pop(counter+1)

temp = [np.round(x, 1) for x in temp]

ax[0, 0].get_xticklabels()[counter].set_color("red")
ax[0, 0].set_xticks(ticks=temp, labels=temp)

ax[0, 0].legend(loc="upper right")
ax[0, 0].set_title("{} - Homogeneity - {}".format(datasets_dict["dataset_name"], method_name))
#ax[0, 0].set_xlabel("epsilon distances")
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

ax[0, 1].get_xticklabels()[counter].set_color("red")
ax[0, 1].set_xticks(ticks=temp, labels=temp)

ax[0, 1].legend(loc="upper right")
ax[0, 1].set_title("{} - NMI - {}".format(datasets_dict["dataset_name"], method_name))
#ax[0, 1].set_xlabel("epsilon distances")
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

ax[1, 0].get_xticklabels()[counter].set_color("red")
ax[1, 0].set_xticks(ticks=temp, labels=temp)

ax[1, 0].legend(loc="upper right")
ax[1, 0].set_title("{} - Rand - {}".format(datasets_dict["dataset_name"], method_name))
ax[1, 0].set_xlabel("epsilon distances")
ax[1, 0].set_ylabel("Rand score")
#ax[0, 0].show()

# *******************************************************
#fig, ax = plt.subplots()
ax[1, 1].plot(
        datasets_dict["distances_interval"],
        V_measure_classic,
        "r--", label="Classic")
ax[1, 1].plot(
        datasets_dict["distances_interval"],
        V_measure_d0,
        "b--", label="d0-method")


ax[1, 1].get_xticklabels()[counter].set_color("red")
ax[1, 1].set_xticks(ticks=temp, labels=temp)

ax[1, 1].legend(loc="upper right")
ax[1, 1].set_title("{} - Vmeasure - {}".format(datasets_dict["dataset_name"], method_name))
ax[1, 1].set_ylabel("Vmeasure score")
ax[1, 1].set_xlabel("epsilon distances")
plt.show()
