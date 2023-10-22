import csv
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from loadDatasets import load_datasets
import os

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")

# 1 - Write each dataset in each own csv file
# 2 - or aggregate all dataset results in one csv file
csv_formatting_flag = 2
if csv_formatting_flag == 2:
    f = open(
        '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/AgglomerativeClustering/Agglomerative evaluation experiments', 'w')
    writer = csv.writer(f)
    writer.writerow(
        ["Dataset name", "Number of labels", "Dimensions", "Number of points", "Homogeneity classic", "Homogeneity d0",
         "Vmeasure classic", "Vmeasure d0", "RAND classic", "RAND d0", "AMI classic",
         "AMI d0", "NMI classic", "NMI d0", "F1 classic", "F1 d0", "Silhouette coefficient classic",
         "Silhouette coefficient d0"])

for dataset in mylist:

    dataset_name = dataset
    datasets_dict = load_datasets(dataset_name)
    numberOfLabels = max(datasets_dict["labels"]) + 1

    # Classic Kmedoids classic:
    db_classic = AgglomerativeClustering(n_clusters=numberOfLabels, linkage='complete').fit(datasets_dict["data"])
    db_classic_labels_pred = db_classic.labels_
    # D0 Kmedoids:

    db_d0 = AgglomerativeClustering(n_clusters=numberOfLabels, affinity='precomputed', linkage='complete').fit(datasets_dict["d0_distances"])
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
        f = open('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/AgglomerativeClustering/{}-AgglomerativeClustering'.format(dataset_name), 'w')
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(["Number of labels", "Homogeneity classic", "Homogeneity d0", "Vmeasure classic", "Vmeasure d0", "RAND classic", "RAND d0", "AMI classic",
                         "AMI d0", "NMI classic", "NMI d0", "F1 classic", "F1 d0", "Silhouette coefficient classic", "Silhouette coefficient d0"])

        writer.writerow([numberOfLabels, homo_classic, homo_d0, vmeasure_classic, vmeasure_d0, rand_classic, rand_d0, ami_classic, ami_d0, nmi_classic, nmi_d0,
                                     f1_classic, f1_d0,metrics.silhouette_score(datasets_dict["data"], db_classic_labels_pred),
                                     metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")])

        f.close()
    else:
        writer.writerow(
            [dataset_name, numberOfLabels, datasets_dict["data"].shape[1], datasets_dict["data"].shape[0], homo_classic,
             homo_d0, vmeasure_classic, vmeasure_d0, rand_classic, rand_d0, ami_classic,
             ami_d0, nmi_classic, nmi_d0,
             f1_classic, f1_d0, metrics.silhouette_score(datasets_dict["data"], db_classic_labels_pred),
             metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")])

f.close()