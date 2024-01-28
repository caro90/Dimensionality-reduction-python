from sklearn.cluster import OPTICS
from sklearn import metrics
from loadDatasets import *
from sklearn.metrics import f1_score
import os
import csv
from density_based_clustering_plotting import *

# Testing several datasets using DBscan clustering
# then evaluating the result using clustering evaluation metrics

def evalMeasures(dataset_name, method_name, customTicking):

    datasets_dict = load_datasets(dataset_name)
    X = datasets_dict["data"]

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

    if not os.path.exists('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/'
                          'default cost function median/{}'.format(method_name)):
        os.mkdir('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/'
                 'default cost function median/{}'.format(method_name))

    if not os.path.exists('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function median/{}/{}'
                                  .format(method_name,dataset_name)):
        os.mkdir('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function median/{}/{}'
                 .format(method_name, dataset_name))

    # open the file in the write mode
    f = open('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function median/{}/{}/{}'
             .format(method_name, dataset_name, dataset_name + "silhouetteCoefficient"), 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["MinPts", "Silhouette coefficient classic", "Silhouette coefficient d0"])

    # Adding d_best to the list of epsilon values that we use for the density based clustering algorithms
    epsilon_values = datasets_dict["distances_interval"].tolist()
    epsilon_values.append(datasets_dict["d_best"].item())
    epsilon_values.sort()

    minPTS_range = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    for j in minPTS_range:

        # Classic OPTICS approach:
        if j > X.shape[0]:
            # min_samples must be no greater than the number of samples in OPTICS algorithm
            break

        optics_classic = OPTICS(min_samples=j).fit(datasets_dict["data"])
        db_classic_labels_pred = optics_classic.labels_

        # d0 OPTICS:
        optics_d0 = OPTICS(eps=datasets_dict["d_best"], min_samples=j, metric="precomputed").fit(
            datasets_dict["d0_distances"])
        db_d0_labels_pred = optics_d0.labels_

        if len(set(db_classic_labels_pred)) >= 2 and len(set(db_d0_labels_pred)) >= 2:
            # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

            # Saving Silhouette score in a csv file:
            writer.writerow([j, metrics.silhouette_score(X, db_classic_labels_pred),
                             metrics.silhouette_score(datasets_dict["d0_distances"],
                             db_d0_labels_pred, metric="precomputed")])

        db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
        db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))

        AMI_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
        AMI_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))

        RAND_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
        RAND_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))

        V_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
        V_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))

        f1_classic.append(f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted'))
        f1_d0.append(f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted'))


    plotting_figures_OPTICS(datasets_dict, db_classic_homogeneity_score,
                                    db_d0_homogeneity_score,
                                    AMI_classic, AMI_d0, customTicking, method_name, dataset_name,
                                    RAND_index_classic, RAND_index_d0, V_measure_classic, f1_classic,
                                    f1_d0, V_measure_d0, minPTS_range)

    # Close csv file for the silhouette coefficient
    f.close()

if __name__ == '__main__':
    # Method name:
    method_name = "OPTICS"
    # Enable/Disable customTicking on the Y-axis
    customTicking = True

    datasets = ["aggregation", "breast_cancer", "D31", "diabetes",
                "digits", "flame", "genes", "iris", "isolet",
                "moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                "spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine", "coil"]

    for dataset in datasets:
        print(f"dataset: {dataset} started")
        evalMeasures(dataset, method_name, customTicking)
