from sklearn.cluster import OPTICS
from sklearn import metrics
from loadDatasets import *
from sklearn.metrics import f1_score
import os
import csv
from density_based_clustering_plotting import *

# Testing several datasets using DBscan clustering
# then evaluating the result using clustering evaluation metrics
# In the following function we calculate the clustering for a given dataset
# and then we assess its score using several clustering metrics from sklearn. The result
# is plotted in a single plot with 3 figures, (classic, piecewise linear d0, alternative d0
# (where we use an alternative cost definition))

def additional_evalMeasures(dataset_name, method_name, customTicking, version, cost_function, cost_function2,
                            path_cost_function, path_cost_function2):

    # Loading the datasets
    datasets_dict = load_datasets(dataset_name, path_cost_function)
    datasets_dict_exponential = load_datasets(dataset_name, path_cost_function2)

    X = datasets_dict["data"]

    db_classic_homogeneity_score = []
    db_d0_homogeneity_score = []
    db_d0_alternative_homogeneity_score = []

    ami_classic = []
    ami_d0 = []
    ami_d0_alternative = []

    rand_index_classic = []
    rand_index_d0 = []
    rand_index_d0_alternative = []

    v_measure_classic = []
    v_measure_d0 = []
    v_measure_d0_alternative = []

    f1_classic = []
    f1_d0 = []
    f1_d0_alternative = []

    # Save location of the figures:
    if not os.path.exists(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                          f'{cost_function}+{cost_function2}'):
        os.mkdir(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                 f'{cost_function}+{cost_function2}')

    if not os.path.exists(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                          f'{cost_function}+{cost_function2}/{method_name}'):
        os.mkdir(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                 f'{cost_function}+{cost_function2}/{method_name}')

    if not os.path.exists(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                          f'{cost_function}+{cost_function2}/{method_name}/{dataset_name}'):
        os.mkdir(
            f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{cost_function}+{cost_function2}/'
            f'{method_name}/{dataset_name}')

    # open the file in the write mode
    f = open(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{cost_function}+{cost_function2}/'
             f'{method_name}/{dataset_name}/{dataset_name + "silhouetteCoefficient"}', 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["MinPts", "Silhouette coefficient classic", "Silhouette coefficient d0",
                     "Silhouette coefficient alternative cost d0"])

    # Adding d_best to the list of epsilon values that we use for the density based clustering algorithms
    epsilon_values = datasets_dict["distances_interval"].tolist()
    epsilon_values.append(datasets_dict["d_best"].item())
    epsilon_values.sort()

    min_pts_range = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240]
    for j in min_pts_range:

        # Classic OPTICS approach:
        if j > X.shape[0]:
            # min_samples must be no greater than the number of samples in OPTICS algorithm
            new_slice = slice(0, min_pts_range.index(j))
            min_pts_range = min_pts_range[new_slice]
            break

        optics_classic = OPTICS(min_samples=j).fit(datasets_dict["data"])
        db_classic_labels_pred = optics_classic.labels_

        # d0 OPTICS:
        optics_d0 = OPTICS(eps=datasets_dict["d_best"], min_samples=j, metric="precomputed").fit(
            datasets_dict["d0_distances"])
        db_d0_labels_pred = optics_d0.labels_

        # d0 alternative cost OPTICS:
        db_d0_alternative = OPTICS(eps=datasets_dict_exponential["d_best"], min_samples=j, metric="precomputed").fit(
            datasets_dict_exponential["d0_distances"])
        db_d0_alternative_labels_pred = db_d0_alternative.labels_


        if (len(set(db_classic_labels_pred)) >= 2 and
            len(set(db_d0_labels_pred)) >= 2 and
            len(set(db_d0_alternative_labels_pred)) >= 2):
            # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

            # Saving Silhouette score in a csv file:
            writer.writerow([j, metrics.silhouette_score(X, db_classic_labels_pred),
                             metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred,
                                                      metric="precomputed"),
                             metrics.silhouette_score(datasets_dict_exponential["d0_distances"],
                                                      db_d0_alternative_labels_pred, metric="precomputed")])

        db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
        db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))
        db_d0_alternative_homogeneity_score.append(
            metrics.homogeneity_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

        ami_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
        ami_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))
        ami_d0_alternative.append(
            metrics.adjusted_mutual_info_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

        rand_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
        rand_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))
        rand_index_d0_alternative.append(metrics.rand_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

        v_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
        v_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))
        v_measure_d0_alternative.append(metrics.v_measure_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

        f1_classic.append(f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted'))
        f1_d0.append(f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted'))
        f1_d0_alternative.append(f1_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred, average='weighted'))

    plotting_figures_OPTICS(datasets_dict, db_classic_homogeneity_score, db_d0_homogeneity_score,
                            db_d0_alternative_homogeneity_score,
                            ami_classic, ami_d0, ami_d0_alternative,
                            customTicking, method_name, dataset_name,
                            rand_index_classic,
                            rand_index_d0,
                            rand_index_d0_alternative, v_measure_classic,
                            f1_classic, f1_d0, f1_d0_alternative, v_measure_d0,
                            v_measure_d0_alternative, min_pts_range, version, f"{cost_function}+{cost_function2}")

    # Close csv file for the silhouette coefficient
    f.close()

if __name__ == '__main__':

    # Method name, use DBSCAN or CommonNN:
    method_name = "OPTICS"

    Version = "Version 0.5"

    # Select the two cost function to be compared against the classic
    cost_function = "default cost"
    cost_function2 = "cost 4"

    # Enable/Disable customTicking on the Y-axis
    customTicking = True

    datasets = ["aggregation", "breast_cancer", "coil", "D31", "diabetes",
                "digits", "flame", "genes", "iris","isolet",
                "moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                "spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]

    for dataset in datasets:
        print(f"dataset: {dataset} started")

        # Setting the right path to load the right precomputed datasets
        path_cost_function = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{Version}/' \
                             f'd0_distances_sin_method/{cost_function}/' \
                             f'20 percent dijkstra points - 100 percent backtracking points - lambda 10000/' \
                             f'{dataset}_d0_distances.mat'

        path_cost_function2 = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{Version}/' \
                              f'd0_distances_sin_method/{cost_function2}/' \
                              f'20 percent dijkstra points - 100 percent backtracking points/' \
                              f'{dataset}_d0_distances.mat'

        additional_evalMeasures(dataset, method_name, customTicking, Version, cost_function,
                                cost_function2, path_cost_function, path_cost_function2)
