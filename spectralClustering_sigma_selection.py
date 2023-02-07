import csv
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from loadDatasets import load_datasets
import numpy as np

#mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/DBSCAN")
mylist = ["diabetes"]

for dataset in mylist:
    # open the file in the write mode
    f = open(
        '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.2/default cost function/SpectralClustering/{}-SpectralClustering'
        .format(dataset), 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(["Eps", "Number of labels", "Homogeneity d0", "Vmeasure d0", "RAND d0", "AMI d0", "NMI d0",
                     "F1 d0", "Silhouette coefficient d0"])

    datasets_dict = load_datasets(dataset)
    numberOfLabels = max(datasets_dict["labels"]) + 1
    for eps in datasets_dict["distances_interval"]:
        print(f"eps:{eps}")
        # d0 Spectral clustering:
        sigma = eps
        d0_distances = np.exp(-datasets_dict["d0_distances"]**2 / (2 * sigma ** 2))
        db_d0 = SpectralClustering(n_clusters=numberOfLabels, assign_labels='discretize', random_state=0, affinity='precomputed', eigen_solver='amg').fit(
            d0_distances)
        db_d0_labels_pred = db_d0.labels_

        homo_d0 = metrics.homogeneity_score(datasets_dict["labels"], db_d0_labels_pred)
        vmeasure_d0 = metrics.v_measure_score(datasets_dict["labels"], db_d0_labels_pred)
        rand_d0 = metrics.rand_score(datasets_dict["labels"], db_d0_labels_pred)
        ami_d0 = metrics.adjusted_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred)
        nmi_d0 = metrics.normalized_mutual_info_score(datasets_dict["labels"], db_d0_labels_pred)
        f1_d0 = metrics.f1_score(datasets_dict["labels"], db_d0_labels_pred, average='weighted')

        # Classic Spectral clustering (euclidean distance):
        

        silhouette_d0 = 0
        if len(set(db_d0_labels_pred)) >= 2:
            silhouette_d0 = metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred, metric="precomputed")

        # Saving the results into a csv file
        # *******************************************************


        writer.writerow([eps, numberOfLabels, homo_d0, vmeasure_d0, rand_d0, ami_d0, nmi_d0, f1_d0, silhouette_d0])

    f.close()
