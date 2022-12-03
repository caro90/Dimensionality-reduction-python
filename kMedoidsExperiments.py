import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from loadDatasets import load_datasets

datasets_dict = load_datasets()
# #########################################################


num_of_classes = np.linspace(2, 20, 20, dtype=int)

distances_interval = np.round(num_of_classes)
# Performance measures
db_classic_homogeneity_score = []
db_d0_homogeneity_score = []

NMI_classic = []
NMI_d0 = []

RAND_index_classic = []
RAND_index_d0 = []

V_measure_classic = []
V_measure_d0 = []


for i in num_of_classes:
    print("I am i", i)

    # Classic DBscan classic:
    db_classic = KMedoids(n_clusters=i).fit(datasets_dict["data"])
    db_classic_labels_pred = db_classic.labels_
    # D0 DBscan:
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
        num_of_classes,
        db_classic_homogeneity_score,
        "r--", label="classic")

ax[0, 0].plot(distances_interval,
        db_d0_homogeneity_score,
        "b--", label="d0-method")


t = ['d0']
a = [datasets_dict["d_best"].max()]
temp = num_of_classes.tolist()
temp.append(datasets_dict["d_best"].max())
temp.sort()
temp_label = temp
temp = [int(x) for x in temp]
t = [str(n) for n in temp]

for i in range(len(t)):
    if t[i] == str(int(datasets_dict["d_best"].max())):
        t[i] = 'd0'
        counter = i


#temp.pop(counter-1)
#t.pop(counter-1)

#ax[0, 0].get_xticklabels()[counter].set_color("red")
#ax[0, 0].set_xticks(ticks=temp, labels=t)

ax[0, 0].legend(loc="upper right")
ax[0, 0].set_title("{} - Homogeneity - {}".format(datasets_dict["dataset_name"], "Kmedoids"))
#ax[0, 0].set_xlabel("Number of clusters")
ax[0, 0].set_ylabel("homogeneity score")


# *******************************************************

ax[0, 1].plot(
        num_of_classes,
        NMI_classic,
        "r--", label="classic")
ax[0, 1].plot(
        num_of_classes,
        NMI_d0,
        "b--", label="d0-method")

#ax[0, 1].get_xticklabels()[counter].set_color("red")
#ax[0, 1].set_xticks(ticks=temp, labels=t)

ax[0, 1].legend(loc="upper right")
ax[0, 1].set_title("{} - NMI - {}".format(datasets_dict["dataset_name"], "Kmedoids"))
#ax[0, 1].set_xlabel("Number of clusters")
ax[0, 1].set_ylabel("NMI score")

# *******************************************************


ax[1, 0].plot(
        num_of_classes,
        RAND_index_classic,
        "r--", label="classic")
ax[1, 0].plot(
        num_of_classes,
        RAND_index_d0,
        "b--", label="d0-method")

#ax[1, 0].get_xticklabels()[counter].set_color("red")
#ax[1, 0].set_xticks(ticks=temp, labels=t)

ax[1, 0].legend(loc="upper right")
ax[1, 0].set_title("{} - Rand - {}".format(datasets_dict["dataset_name"], "Kmedoids"))
ax[1, 0].set_xlabel("Number of clusters")
ax[1, 0].set_ylabel("Rand score")
#ax[0, 0].show()

# *******************************************************
#fig, ax = plt.subplots()
ax[1, 1].plot(
        num_of_classes,
        V_measure_classic,
        "r--", label="Classic")
ax[1, 1].plot(
        num_of_classes,
        V_measure_d0,
        "b--", label="d0-method")


#ax[1, 1].get_xticklabels()[counter].set_color("red")
#ax[1, 1].set_xticks(ticks=temp, labels=t)

ax[1, 1].legend(loc="upper right")
ax[1, 1].set_title("{} - Vmeasure - {}".format(datasets_dict["dataset_name"], "Kmedoids"))
ax[1, 1].set_ylabel("Vmeasure score")
ax[1, 1].set_xlabel("Number of clusters")
plt.show()