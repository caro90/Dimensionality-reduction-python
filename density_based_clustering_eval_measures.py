from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn_extra.cluster import CommonNNClustering
from loadDatasets import *
from sklearn.metrics import f1_score
import os
import csv
from density_based_clustering_plotting import *

def additional_evalMeasures(dataset_name, method_name, customTicking, version, cost_function, cost_function2,
                            path_cost_function, path_cost_function2):

    # Testing several datasets using DBscan and CommonNN clustering
    # then evaluating the result using clustering evaluation metrics

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
        os.mkdir(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{cost_function}+{cost_function2}/'
                 f'{method_name}/{dataset_name}')

    # open the file in the write mode
    f = open(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{cost_function}+{cost_function2}/'
             f'{method_name}/{dataset_name}/{dataset_name + "silhouetteCoefficient"}', 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["MinPts", "eps", "Silhouette coefficient classic", "Silhouette coefficient d0", "Silhouette coefficient alternative cost d0"])

    # Adding d_best to the list of epsilon values that we use for the density based clustering algorithms
    epsilon_values = datasets_dict["distances_interval"].tolist()
    epsilon_values.append(datasets_dict["d_best"].item())
    epsilon_values.sort()

    min_pts_range = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240]
    for j in min_pts_range:
        for i in epsilon_values:

            if method_name == "DBSCAN":
                # Classic DBscan approach:
                db_classic = DBSCAN(eps=i, min_samples=j).fit(datasets_dict["data"])
                db_classic_labels_pred = db_classic.labels_

                # d0 DBscan:
                db_d0 = DBSCAN(eps=i, min_samples=j, metric="precomputed").fit(datasets_dict["d0_distances"])
                db_d0_labels_pred = db_d0.labels_

                # d0 alternative cost DBscan:
                db_d0_alternative = DBSCAN(eps=i, min_samples=j, metric="precomputed").fit(datasets_dict_exponential["d0_distances"])
                db_d0_alternative_labels_pred = db_d0_alternative.labels_

            elif method_name == "CommonNN":
                # Classic CommonNN:
                db_classic = CommonNNClustering(eps=i, min_samples=j).fit(datasets_dict["data"])
                db_classic_labels_pred = db_classic.labels_

                # D0 CommonNN:
                db_d0 = CommonNNClustering(eps=i, min_samples=j, metric="precomputed").fit(
                    datasets_dict["d0_distances"])
                db_d0_labels_pred = db_d0.labels_

                # d0 alternative cost CommonNN:
                db_d0_alternative = (CommonNNClustering(eps=i, min_samples=j, metric="precomputed").
                                                    fit(datasets_dict_exponential["d0_distances"]))
                db_d0_alternative_labels_pred = db_d0_alternative.labels_

            if (len(set(db_classic_labels_pred)) >= 2 and
                len(set(db_d0_labels_pred)) and
                len(set(db_d0_alternative_labels_pred)) >= 2):
                # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

                # Saving Silhouette score in a csv file:
                writer.writerow([j, i, metrics.silhouette_score(X, db_classic_labels_pred),
                                 metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed"),
                                 metrics.silhouette_score(datasets_dict_exponential["d0_distances"], db_d0_alternative_labels_pred, metric="precomputed")])

            db_classic_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred))
            db_d0_homogeneity_score.append(metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred))
            db_d0_alternative_homogeneity_score.append(metrics.homogeneity_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

            ami_classic.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred))
            ami_d0.append(metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred))
            ami_d0_alternative.append(metrics.adjusted_mutual_info_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

            rand_index_classic.append(metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred))
            rand_index_d0.append(metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred))
            rand_index_d0_alternative.append(metrics.rand_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

            v_measure_classic.append(metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred))
            v_measure_d0.append(metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred))
            v_measure_d0_alternative.append(metrics.v_measure_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred))

            f1_classic.append(f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted'))
            f1_d0.append(f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted'))
            f1_d0_alternative.append(f1_score(datasets_dict_exponential["labels"], db_d0_alternative_labels_pred, average='weighted'))


        plotting_figures_DBSAN_CommonNN(epsilon_values, datasets_dict, datasets_dict_exponential,
                                        db_classic_homogeneity_score,
                                        db_d0_homogeneity_score,
                                        db_d0_alternative_homogeneity_score,
                                        ami_classic,
                                        ami_d0,
                                        ami_d0_alternative,
                                        customTicking, method_name, dataset_name,
                                        rand_index_classic,
                                        rand_index_d0,
                                        rand_index_d0_alternative,
                                        v_measure_classic, f1_classic,
                                        f1_d0, f1_d0_alternative,
                                        v_measure_d0,
                                        v_measure_d0_alternative,
                                        j, version, f"{cost_function}+{cost_function2}")

        # Clear the lists for the next run
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

    # Close csv file for the silhouette coefficient
    f.close()

if __name__ == '__main__':

    # Method name, use DBSCAN or CommonNN:
    method_name = "CommonNN"

    Version = "Version 0.5"

    # Select the two cost function to be compared against the classic
    cost_function = "default cost"
    cost_function2 = "cost 4"

    # Enable/Disable customTicking on the Y-axis
    customTicking = True

    #"aggregation", "breast_cancer", "coil", "D31", "diabetes",
                #"digits", "flame", "genes", "iris", "isolet",
                #"moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                #"spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"

    datasets = ["wine"]

    for dataset in datasets:
        print(f"dataset: {dataset} started")

        # Setting the right path to load the right precomputed datasets
        path_cost_function = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{Version}/'\
                             f'd0_distances_sin_method/{cost_function}/'\
                             f'20 percent dijkstra points - 100 percent backtracking points - lambda 10000/'\
                             f'{dataset}_d0_distances.mat'

        path_cost_function2 = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{Version}/'\
                              f'd0_distances_sin_method/{cost_function2}/'\
                              f'20 percent dijkstra points - 100 percent backtracking points/'\
                              f'{dataset}_d0_distances.mat'

        additional_evalMeasures(dataset, method_name, customTicking, Version, cost_function,
                                cost_function2, path_cost_function, path_cost_function2)
