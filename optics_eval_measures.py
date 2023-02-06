# Testing several datasets using OPTICS clustering
# then evaluating the result using clustering evaluation metrics
from sklearn.cluster import OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import *
from sklearn.metrics import f1_score
import os
import pickle
import csv

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/DBSCAN")

db_classic_homogeneity_score = []
db_d0_homogeneity_score = []

AMI_classic = []
AMI_d0 = []

RAND_index_classic = []
RAND_index_d0 = []

V_measure_classic = []
V_measure_d0 = []

f1_classic = []
f1_d0 = []

for dataset in mylist:
    min_pts_list = [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80]
    if not os.path.exists(
            '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}'
                    .format(dataset)):
        os.mkdir('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}'
                 .format(dataset))
    datasets_dict = load_datasets(dataset)
    print("{}".format(dataset))

    # open the file in the write mode for silhouetteCoefficient
    f = open(
        '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}/{}'
        .format(dataset, dataset + "silhouetteCoefficient"), 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["MinPts", "Silhouette coefficient classic", "Silhouette coefficient d0"])

    for j in min_pts_list:
        # Classic OPTICS approach:
        optics_classic = OPTICS(min_samples=j).fit(datasets_dict["data"])
        optics_classic_labels_pred = optics_classic.labels_

        # d0 OPTICS:
        optics_d0 = OPTICS(eps=datasets_dict["d_best"], min_samples=j, metric="precomputed").fit(datasets_dict["d0_distances"])
        optics_d0_labels_pred = optics_d0.labels_

        if len(set(optics_classic_labels_pred)) >= 2 and len(set(optics_d0_labels_pred)) >= 2:
            # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

            # Saving Silhouette score in a csv file:
            writer.writerow([j, metrics.silhouette_score(datasets_dict["data"], optics_classic_labels_pred),
                             metrics.silhouette_score(datasets_dict["d0_distances"], optics_d0_labels_pred, metric="precomputed")])

        db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], optics_classic_labels_pred))
        db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], optics_d0_labels_pred))

        AMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], optics_classic_labels_pred))
        AMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], optics_d0_labels_pred))

        RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], optics_classic_labels_pred))
        RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], optics_d0_labels_pred))

        V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], optics_classic_labels_pred))
        V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], optics_d0_labels_pred))

        f1_classic.append(f1_score(datasets_dict["labels"], optics_classic_labels_pred, average='weighted'))
        f1_d0.append(f1_score(datasets_dict["labels"], optics_d0_labels_pred, average='weighted'))

    # Plotting
    fig, ax = plt.subplots(1, 2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        min_pts_list,
        db_classic_homogeneity_score,
        "r--", label="classic")

    ax[0].plot(min_pts_list,
                  db_d0_homogeneity_score,
                  "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Homogeneity")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        AMI_classic,
        "r--", label="classic")
    ax[1].plot(
        min_pts_list,
        AMI_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("AMI")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}/{}-OPTICS(Homogeneity,AMI)-default cost"\
        .format(dataset, dataset)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    # *******************************************************
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        min_pts_list,
        RAND_index_classic,
        "r--", label="classic")
    ax[0].plot(
        min_pts_list,
        RAND_index_d0,
        "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Rand")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        V_measure_classic,
        "r--", label="Classic")
    ax[1].plot(
        min_pts_list,
        V_measure_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}/{}-OPTICS(Vmeasure,RAND)-default cost"\
        .format(dataset, dataset)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1)
    plt.plot(
        min_pts_list,
        f1_classic,
        "r--", label="classic")
    plt.plot(
        min_pts_list,
        f1_d0,
        "b--", label="d0-method")

    plt.legend(loc="upper right")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))
    plt.title("F1 score")
    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/OPTICS/{}/{}-OPTICS(f1)-default cost"\
        .format(dataset, dataset)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    # Clear the lists for the next run
    db_classic_homogeneity_score = []
    db_d0_homogeneity_score = []

    AMI_classic = []
    AMI_d0 = []

    RAND_index_classic = []
    RAND_index_d0 = []

    V_measure_classic = []
    V_measure_d0 = []

    f1_classic = []
    f1_d0 = []

    # Close csv file for the silhouette coefficient
    f.close()