import csv
from sklearn import metrics
from sklearn_extra.cluster import KMedoids
from loadDatasets import load_datasets
import os

# Testing several datasets using Kmedoids clustering
# then evaluating the result using clustering evaluation metrics
# In the following function we calculate the clustering for a given dataset
# and then we assess its score using several clustering metrics from sklearn. The result
# is plotted in a single plot with 3 figures, (classic, piecewise linear d0, alternative d0
# (where we use an alternative cost definition))


def additional_eval_measures(dataset_name, writer, path_cost_function, path_cost_function2):

    datasets_dict1 = load_datasets(dataset_name, path_cost_function)
    datasets_dict2 = load_datasets(dataset_name, path_cost_function2)

    number_of_labels = max(datasets_dict1["labels"]) + 1

    # Classic Kmedoids classic:
    db_classic = KMedoids(number_of_labels).fit(datasets_dict1["data"])
    db_classic_labels_pred = db_classic.labels_

    # D0 Kmedoids:
    db_d0 = KMedoids(number_of_labels, metric="precomputed").fit(datasets_dict1["d0_distances"])
    db_d0_labels_pred = db_d0.labels_

    # Alternative D0 Kmedoids:
    db_d0_alternative = KMedoids(number_of_labels, metric="precomputed").fit(datasets_dict2["d0_distances"])
    db_d0_alternative_labels_pred = db_d0_alternative.labels_

    # Clustering metrics:
    homo_classic = metrics.homogeneity_score(datasets_dict1["labels"], db_classic_labels_pred)
    homo_d0 = metrics.homogeneity_score(datasets_dict1["labels"], db_d0_labels_pred)
    homo_d0_alternative = metrics.homogeneity_score(datasets_dict2["labels"], db_d0_alternative_labels_pred)

    vmeasure_classic = metrics.v_measure_score(datasets_dict1["labels"], db_classic_labels_pred)
    vmeasure_d0 = metrics.v_measure_score(datasets_dict1["labels"], db_d0_labels_pred)
    vmeasure_d0_alternative = metrics.v_measure_score(datasets_dict2["labels"], db_d0_alternative_labels_pred)

    rand_classic = metrics.rand_score(datasets_dict1["labels"], db_classic_labels_pred)
    rand_d0 = metrics.rand_score(datasets_dict1["labels"], db_d0_labels_pred)
    rand_d0_alternative = metrics.rand_score(datasets_dict2["labels"], db_d0_alternative_labels_pred)

    ami_classic = metrics.adjusted_mutual_info_score(datasets_dict1["labels"], db_classic_labels_pred)
    ami_d0 = metrics.adjusted_mutual_info_score(datasets_dict1["labels"], db_d0_labels_pred)
    ami_d0_alternative = metrics.adjusted_mutual_info_score(datasets_dict2["labels"], db_d0_alternative_labels_pred)

    nmi_classic = metrics.normalized_mutual_info_score(datasets_dict1["labels"], db_classic_labels_pred)
    nmi_d0 = metrics.normalized_mutual_info_score(datasets_dict1["labels"], db_d0_labels_pred)
    nmi_d0_alternative = metrics.normalized_mutual_info_score(datasets_dict2["labels"], db_d0_alternative_labels_pred)

    f1_classic = metrics.f1_score(datasets_dict1["labels"], db_classic_labels_pred, average='weighted')
    f1_d0 = metrics.f1_score(datasets_dict1["labels"], db_d0_labels_pred, average='weighted')
    f1_d0_alternative = metrics.f1_score(datasets_dict2["labels"], db_d0_alternative_labels_pred, average='weighted')

    # Saving the results into a csv file:
    writer.writerow(
        [dataset_name, number_of_labels, datasets_dict1["data"].shape[1], datasets_dict1["data"].shape[0],
         datasets_dict1["d_best"][0][0], datasets_dict2["d_best"][0][0],
         homo_classic, homo_d0, homo_d0_alternative,
         vmeasure_classic, vmeasure_d0, vmeasure_d0_alternative,
         rand_classic, rand_d0, rand_d0_alternative,
         ami_classic, ami_d0, ami_d0_alternative,
         nmi_classic, nmi_d0, nmi_d0_alternative,
         f1_classic, f1_d0, f1_d0_alternative,
         metrics.silhouette_score(datasets_dict1["data"], db_classic_labels_pred),
         metrics.silhouette_score(datasets_dict1["d0_distances"],db_d0_labels_pred, metric="precomputed"),
         metrics.silhouette_score(datasets_dict2["d0_distances"], db_d0_alternative_labels_pred, metric="precomputed")
         ])

if __name__ == '__main__':

    # Method name:
    method_name = "Kmedoids"

    version = "Version 0.5"
    sampling_method = "d0_distances_sin_method"
    num_of_dijkstra_backtracking_points = "20 percent dijkstra points - 100 percent backtracking points"

    # Select the two cost function to be compared against the classic
    cost_function = "default cost"
    cost_function2 = "cost 3"

    datasets = ["aggregation", "breast_cancer", "coil", "D31", "diabetes",
                "digits", "flame", "genes", "iris", "isolet",
                "moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                "spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]

    # Save location of the figures:
    save_loc_folder_path = (f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{sampling_method}/'
                            f'{cost_function}+{cost_function2}/{num_of_dijkstra_backtracking_points}/'
                            f'{method_name}')
    os.makedirs(save_loc_folder_path, exist_ok=True)
    f = open(
        f'{save_loc_folder_path}/kmedoids evaluation experiments', 'w')
    writer = csv.writer(f)
    writer.writerow(
        ["Dataset name", "Number of labels", "Dimensions", "Number of points",
         "piece-wise linear d0", "exponential d0",
         "Homogeneity classic", "Homogeneity d0", "Homogeneity exponential",
         "Vmeasure classic", "Vmeasure d0", "Vmeasure exponential",
         "RAND classic", "RAND d0", "RAND exponential",
         "AMI classic", "AMI d0", "AMI exponential",
         "NMI classic", "NMI d0", "NMI exponential",
         "F1 classic", "F1 d0", "F1 exponential",
         "Silhouette coefficient classic", "Silhouette coefficient d0", "Silhouette coefficient exponential",])

    for dataset in datasets:
        print(f"dataset: {dataset} started")

        # Setting the right path to load the right precomputed datasets
        path_cost_function = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{version}/' \
                             f'{sampling_method}/{cost_function}/' \
                             f'{num_of_dijkstra_backtracking_points} - lambda 10000/' \
                             f'{dataset}_d0_distances.mat'

        path_cost_function2 = f'/home/arch/Matlab/Dimensionality Reduction/mat_files/{version}/' \
                              f'{sampling_method}/{cost_function2}/' \
                              f'{num_of_dijkstra_backtracking_points}/' \
                              f'{dataset}_d0_distances.mat'

        additional_eval_measures(dataset, writer, path_cost_function, path_cost_function2)

    f.close()