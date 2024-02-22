import csv
from sklearn import metrics
from sklearn_extra.cluster import KMedoids
from loadDatasets import load_datasets
import os

def evalMeasures(dataset_name, method_name, version, cost_function, alternative_cost, csv_formatting_flag, writer):

    datasets_dict = load_datasets(dataset_name, alternative_cost, version, cost_function)
    number_of_labels = max(datasets_dict["labels"]) + 1

    # Classic Kmedoids classic:
    db_classic = KMedoids(number_of_labels).fit(datasets_dict["data"])
    db_classic_labels_pred = db_classic.labels_
    # D0 Kmedoids:
    db_d0 = KMedoids(number_of_labels, metric="precomputed").fit(datasets_dict["d0_distances"])
    db_d0_labels_pred = db_d0.labels_

    homo_classic = metrics.homogeneity_score(datasets_dict["labels"], db_classic_labels_pred)
    homo_d0 = metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred)

    vmeasure_classic = metrics.v_measure_score(datasets_dict["labels"], db_classic_labels_pred)
    vmeasure_d0 = metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred)

    rand_classic = metrics.rand_score(datasets_dict["labels"], db_classic_labels_pred)
    rand_d0 = metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred)

    ami_classic = metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred)
    ami_d0 = metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred)

    nmi_classic = metrics.normalized_mutual_info_score(datasets_dict["labels"], db_classic_labels_pred)
    nmi_d0 = metrics.normalized_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred)

    f1_classic = metrics.f1_score(datasets_dict["labels"], db_classic_labels_pred, average='weighted')
    f1_d0 = metrics.f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted')

    # Saving the results into a csv file
    # *******************************************************

    if csv_formatting_flag == 1:
        # open the file in the write mode
        f = open(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                 f'{cost_function}/{method_name}/{dataset_name}-kmedoids', 'w')
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(["Number of labels", "Dimensions", "Number of points","Homogeneity classic", "Homogeneity d0",
                         "Vmeasure classic", "Vmeasure d0", "RAND classic", "RAND d0", "AMI classic",
                         "AMI d0", "NMI classic", "NMI d0", "F1 classic", "F1 d0", "Silhouette coefficient classic",
                         "Silhouette coefficient d0"])

        writer.writerow([number_of_labels, homo_classic, homo_d0, vmeasure_classic, vmeasure_d0, rand_classic, rand_d0,
                         ami_classic, ami_d0, nmi_classic, nmi_d0, f1_classic, f1_d0,
                         metrics.silhouette_score(datasets_dict["data"], db_classic_labels_pred),
                         metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")])
        f.close()
    else:
        writer.writerow(
            [dataset_name, number_of_labels, datasets_dict["data"].shape[1], datasets_dict["data"].shape[0],
             homo_classic, homo_d0, vmeasure_classic, vmeasure_d0, rand_classic, rand_d0, ami_classic,
             ami_d0, nmi_classic, nmi_d0, f1_classic, f1_d0, metrics.silhouette_score(datasets_dict["data"],
             db_classic_labels_pred), metrics.silhouette_score(datasets_dict["d0_distances"],
             db_d0_labels_pred, metric="precomputed")])




if __name__ == '__main__':

    # Method name:
    method_name = "Kmedoids"
    version = "Version 0.5"
    cost_function = "cost 4"

    # Is the cost other than the default?:
    alternative_cost = True

    # 1 - Write each dataset in each own csv file
    # 2 - or aggregate all dataset results in one csv file
    csv_formatting_flag = 2

    datasets = ["aggregation", "breast_cancer", "coil", "D31", "diabetes",
                "digits", "flame", "genes", "iris", "isolet",
                "moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                "spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]

    if not os.path.exists(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                          f'{cost_function}/{method_name}'):
        os.mkdir(f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/'
                 f'{cost_function}/{method_name}')
    if csv_formatting_flag == 2:
        f = open(
            f'/home/arch/PycharmProjects/Dimensionality reduction results/{version}/{cost_function}/'
            f'{method_name}/kmedoids evaluation experiments', 'w')
        writer = csv.writer(f)
        writer.writerow(
            ["Dataset name", "Number of labels", "Dimensions", "Number of points", "Homogeneity classic",
             "Homogeneity d0",
             "Vmeasure classic", "Vmeasure d0", "RAND classic", "RAND d0", "AMI classic",
             "AMI d0", "NMI classic", "NMI d0", "F1 classic", "F1 d0", "Silhouette coefficient classic",
             "Silhouette coefficient d0"])

    for dataset in datasets:
        print(f"dataset: {dataset} started")
        evalMeasures(dataset, method_name, version, cost_function, alternative_cost, csv_formatting_flag, writer)

    f.close()