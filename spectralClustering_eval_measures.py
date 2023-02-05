import csv
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from loadDatasets import load_datasets
import os
import numpy as np

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/DBSCAN")

for dataset in mylist:

    dataset_name = dataset
    datasets_dict = load_datasets(dataset_name)
    numberOfLabels = max(datasets_dict["labels"]) + 1

    # Classic Spectral clustering classic:
    db_classic = SpectralClustering(n_clusters=numberOfLabels, assign_labels='discretize', random_state=0).fit(datasets_dict["data"])
    db_classic_labels_pred = db_classic.labels_

    # d0 Spectral clustering:
    d0_distances = np.exp(-datasets_dict["d0_distances"]**2 / (2 * datasets_dict["d_best"]**2))
    db_d0 = SpectralClustering(n_clusters=numberOfLabels, assign_labels='discretize', random_state=0, affinity='precomputed').fit(
        d0_distances)
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

    if len(set(db_classic_labels_pred)) >= 2 and len(set(db_d0_labels_pred)) >= 2:
        silhouette_classic = metrics.silhouette_score(datasets_dict["data"], db_classic_labels_pred)
        silhouette_d0 = metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")
    else:
        silhouette_classic = 0
        silhouette_d0 = 0
    # Saving the results into a csv file
    # *******************************************************

    # open the file in the write mode
    f = open('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/SpectralClustering/{}-SpectralClustering'
             .format(dataset_name), 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["Number of labels", "Homogeneity classic", "Homogeneity d0", "Vmeasure classic", "Vmeasure d0", "RAND classic", "RAND d0", "AMI classic",
                     "AMI d0", "NMI classic", "NMI d0", "F1 classic", "F1 d0", "Silhouette coefficient classic", "Silhouette coefficient d0"])

    writer.writerow([numberOfLabels, homo_classic, homo_d0, vmeasure_classic, vmeasure_d0, rand_classic, rand_d0, ami_classic, ami_d0, nmi_classic, nmi_d0,
                                 f1_classic, f1_d0, silhouette_classic, silhouette_d0])

    f.close()


#Chat GPT response on how to calculate the affinity matrix

# from sklearn.cluster import SpectralClustering
# import numpy as np
#
# # create a distance matrix
# D = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]])
#
# # define sigma
# sigma = 1
#
# # calculate the affinity matrix
# A = np.exp(-D**2/(2*sigma**2))
#
# # perform spectral clustering
# sc = SpectralClustering(n_clusters=2)
# clusters = sc.fit_predict(A)