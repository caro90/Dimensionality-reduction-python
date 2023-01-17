# Testing several datasets using DBscan clustering
# then evaluating the result using clustering evaluation metrics

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import *
from sklearn.metrics import f1_score
import numpy as np
import os
import pickle
import csv

dataset_name = "wine"
method_name = "OPTICS"

datasets_dict = load_datasets(dataset_name)

X = datasets_dict["data"]

db_classic_homogeneity_score = []
db_d0_homogeneity_score = []

NMI_classic = []
NMI_d0 = []

RAND_index_classic = []
RAND_index_d0 = []

V_measure_classic = []
V_measure_d0 = []

f1_classic = []
f1_d0 = []

if not os.path.exists('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}'.format(method_name,
                dataset_name)):
    os.mkdir('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}'.format(method_name,
        dataset_name))

# open the file in the write mode
f = open('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}/{}'.format(method_name, dataset_name, dataset_name + "silhouetteCoefficient"), 'w')
# create the csv writer
writer = csv.writer(f)
writer.writerow(["MinPts", "eps", "Silhouette coefficient classic", "Silhouette coefficient d0"])

min_pts_list = [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80]
for j in min_pts_list:
    print("MinPts: {}".format(j))
    for i in datasets_dict["distances_interval"]:

        if method_name == "DBSCAN":
            # Classic DBscan approach:
            db_classic = DBSCAN(eps=i, min_samples=j).fit(datasets_dict["data"])
            db_classic_labels_pred = db_classic.labels_

            # d0 DBscan:
            db_d0 = DBSCAN(eps=i, min_samples=j, metric="precomputed").fit(datasets_dict["d0_distances"])
            db_d0_labels_pred = db_d0.labels_

        if method_name == "OPTICS":
            db_classic = OPTICS(eps=i, min_samples=j).fit(datasets_dict["data"])
            db_classic_labels_pred = db_classic.labels_
            # d0 OPTICS:
            db_d0 = OPTICS(eps=i, min_samples=j, metric="precomputed").fit(datasets_dict["d0_distances"])
            db_d0_labels_pred = db_d0.labels_

        if len(set(db_classic_labels_pred)) >= 2 and len(set(db_d0_labels_pred)) >= 2:
            # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

            # Saving Silhouette score in a csv file:
            writer.writerow([j, i, metrics.silhouette_score(X, db_classic_labels_pred),
                             metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")])

        db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
        db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))

        NMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
        NMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))

        RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
        RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))

        V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
        V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))

        f1_classic.append(f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted'))
        f1_d0.append(f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted'))

    # Plotting
    fig, ax = plt.subplots(1, 2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        datasets_dict["distances_interval"],
        db_classic_homogeneity_score,
        "r--", label="classic")

    ax[0].plot(datasets_dict["distances_interval"],
                  db_d0_homogeneity_score,
                  "b--", label="d0-method")

    t = ['d0']
    a = [datasets_dict["d_best"].max()]
    temp = datasets_dict["distances_interval"].tolist()
    temp.append(datasets_dict["d_best"].max())
    temp.sort()

    for i in range(0, len(temp)):
        if temp[i] == datasets_dict["d_best"].max():
            counter = i

    temp = [np.round(x, 1) for x in temp]

    ax[0].legend(loc="upper right")
    ax[0].set_title("Homogeneity")
    ax[0].set_xlabel("epsilon distances")

    # *******************************************************
    ax[1].plot(
        datasets_dict["distances_interval"],
        NMI_classic,
        "r--", label="classic")
    ax[1].plot(
        datasets_dict["distances_interval"],
        NMI_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("NMI")
    ax[1].set_xlabel("epsilon distances")
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}/{}-DBSCAN(Homogeneity,NMI)-default cost-Min_pts {}"\
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    # *******************************************************
    fig, ax = plt.subplots(1, 2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        datasets_dict["distances_interval"],
        RAND_index_classic,
        "r--", label="classic")
    ax[0].plot(
        datasets_dict["distances_interval"],
        RAND_index_d0,
        "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Rand")
    ax[0].set_xlabel("epsilon distances")

    # *******************************************************
    ax[1].plot(
        datasets_dict["distances_interval"],
        V_measure_classic,
        "r--", label="Classic")
    ax[1].plot(
        datasets_dict["distances_interval"],
        V_measure_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ".format(datasets_dict["dataset_name"], method_name))
    ax[1].set_xlabel("epsilon distances")
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}/{}-DBSCAN(Vmeasure,RAND)-default cost-Min_pts {}"\
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1)
    plt.plot(
        datasets_dict["distances_interval"],
        f1_classic,
        "r--", label="classic")
    plt.plot(
        datasets_dict["distances_interval"],
        f1_d0,
        "b--", label="d0-method")

    plt.legend(loc="upper right")
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))
    plt.title("F1 score ")

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/{}/{}/{}-DBSCAN(f1)-default cost-Min_pts {}"\
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    # Clear the lists for the next run
    db_classic_homogeneity_score = []
    db_d0_homogeneity_score = []

    NMI_classic = []
    NMI_d0 = []

    RAND_index_classic = []
    RAND_index_d0 = []

    V_measure_classic = []
    V_measure_d0 = []

    f1_classic = []
    f1_d0 = []

# Close csv file for the silhouette coefficient
f.close()