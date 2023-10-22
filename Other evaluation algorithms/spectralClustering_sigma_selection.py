import csv
import os
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from loadDatasets import load_datasets
import numpy as np

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"
                    "DBSCAN")
# create a main function

for dataset in mylist:
    print(f"dataset name:{dataset}")
    # open the file in the write mode
    f = open(
        '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/'
        'SpectralClustering/{}-SpectralClustering-d0-sigmaSelection'
        .format(dataset), 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["Eps", "Number of labels", "Homogeneity d0", "Vmeasure d0", "RAND d0", "AMI d0", "NMI d0",
                     "F1 d0", "Silhouette coefficient d0"])

    # open the file in the write mode
    f2 = open(
        '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/'
        'SpectralClustering/{}-SpectralClustering-euclidean-sigmaSelection'
        .format(dataset), 'w')
    # create the csv writer
    writer2 = csv.writer(f2)
    writer2.writerow(["Eps", "Number of labels", "Homogeneity", "Vmeasure", "RAND", "AMI", "NMI",
                     "F1", "Silhouette coefficient"])

    datasets_dict = load_datasets(dataset)
    numberOfLabels = max(datasets_dict["labels"]) + 1
    for eps in datasets_dict["distances_interval"]:

        sigma = eps

        # d0 Spectral clustering:
        d0_distances = np.exp(-datasets_dict["d0_distances"]**2 / (2 * sigma ** 2))
        spectralClustering_d0 = SpectralClustering(n_clusters=numberOfLabels, assign_labels='discretize', random_state=0,
                                                   affinity='precomputed', eigen_solver='amg').fit(d0_distances)
        spectralClustering_d0_labels_pred = spectralClustering_d0.labels_

        homo_d0 = metrics.homogeneity_score(datasets_dict["labels"], spectralClustering_d0_labels_pred)
        vmeasure_d0 = metrics.v_measure_score(datasets_dict["labels"], spectralClustering_d0_labels_pred)
        rand_d0 = metrics.rand_score(datasets_dict["labels"], spectralClustering_d0_labels_pred)
        ami_d0 = metrics.adjusted_mutual_info_score(datasets_dict["labels"], spectralClustering_d0_labels_pred)
        nmi_d0 = metrics.normalized_mutual_info_score(datasets_dict["labels"], spectralClustering_d0_labels_pred)
        f1_d0 = metrics.f1_score(datasets_dict["labels"], spectralClustering_d0_labels_pred, average='weighted')

        # Classic Spectral clustering (euclidean distance):
        classic_distances = np.exp(-datasets_dict["D"] ** 2 / (2 * sigma ** 2))
        spectralClustering_classic = SpectralClustering(n_clusters=numberOfLabels, assign_labels='discretize', random_state=0,
                                                   affinity='precomputed', eigen_solver='amg').fit(classic_distances)
        spectralClustering_classic_labels_pred = spectralClustering_classic.labels_

        homo_classic = metrics.homogeneity_score(datasets_dict["labels"], spectralClustering_classic_labels_pred)
        vmeasure_classic = metrics.v_measure_score(datasets_dict["labels"], spectralClustering_classic_labels_pred)
        rand_classic = metrics.rand_score(datasets_dict["labels"], spectralClustering_classic_labels_pred)
        ami_classic = metrics.adjusted_mutual_info_score(datasets_dict["labels"], spectralClustering_classic_labels_pred)
        nmi_classic = metrics.normalized_mutual_info_score(datasets_dict["labels"], spectralClustering_classic_labels_pred)
        f1_classic = metrics.f1_score(datasets_dict["labels"], spectralClustering_classic_labels_pred, average='weighted')

        silhouette_d0 = 0
        if len(set(spectralClustering_d0_labels_pred)) >= 2:
            silhouette_d0 = metrics.silhouette_score(datasets_dict["d0_distances"], spectralClustering_d0_labels_pred,
                                                     metric="precomputed")

        silhouette_classic = 0
        if len(set(spectralClustering_classic_labels_pred)) >= 2:
            silhouette_classic = metrics.silhouette_score(datasets_dict["D"], spectralClustering_classic_labels_pred,
                                                     metric="precomputed")

        # Saving the results into a csv file
        # *******************************************************
        writer.writerow([eps, numberOfLabels, homo_d0, vmeasure_d0, rand_d0, ami_d0, nmi_d0, f1_d0, silhouette_d0])
        writer2.writerow([eps, numberOfLabels, homo_classic, vmeasure_classic, rand_classic, ami_classic, nmi_classic, f1_classic, silhouette_classic])

    f.close()
    f2.close()
