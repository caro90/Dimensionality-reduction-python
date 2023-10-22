import csv
import pickle
from matplotlib import ticker
from sklearn import metrics
import matplotlib.pyplot as plt
from loadDatasets import load_datasets
from sklearn_extra.cluster import CommonNNClustering
from sklearn.metrics import f1_score
import os
from minPtsSampling import generate_min_samples_range

def customTickingForYaxis(ax, axesCounterList):

    if len(axesCounterList) > 1:
        for axesCounter in axesCounterList:

            ymax1 = max(ax[axesCounter].lines[0].get_ydata())
            ymax2 = max(ax[axesCounter].lines[1].get_ydata())

            ax[axesCounter].yaxis.set_major_locator(ticker.FixedLocator([max(ymax1, ymax2)]))
            ymax = max(ymax1, ymax2)
            for finalYmax in ax[axesCounter].yaxis.get_ticklabels():
                if ymax == ymax1:
                    finalYmax.set_color("red")
                else:
                    finalYmax.set_color("blue")
    else:
        ymax1 = max(ax.lines[0].get_ydata())
        ymax2 = max(ax.lines[1].get_ydata())

        ax.yaxis.set_major_locator(ticker.FixedLocator([max(ymax1, ymax2)]))
        ymax = max(ymax1, ymax2)
        for finalYmax in ax.yaxis.get_ticklabels():
            if ymax == ymax1:
                finalYmax.set_color("red")
            else:
                finalYmax.set_color("blue")

def evalMeasures(datasets, method_name, customTicking):

    for dataset in datasets:

        dataset_name = dataset
        datasets_dict = load_datasets(dataset_name)
        X = datasets_dict["data"]

        # Performance measures
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

        if not os.path.exists('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}'
                                      .format(method_name, dataset_name)):
            os.mkdir('/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}'
                     .format(method_name, dataset_name))

        minPTS_range = generate_min_samples_range(X.shape[1])
        for index, minPTS in enumerate(minPTS_range):
            # open the file in the write mode
            f = open(
                '/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}'
                .format(method_name, dataset_name, dataset_name + "silhouetteCoefficient"), 'a')
            writer = csv.writer(f)
            if index == 0:
                # create the csv writer
                writer.writerow(["minPTS", "eps", "Silhouette coefficient classic", "Silhouette coefficient d0"])

            for i in datasets_dict["distances_interval"]:

                # Classic CommonNN:
                db_classic = CommonNNClustering(eps=i, min_samples=minPTS).fit(datasets_dict["data"])
                db_classic_labels_pred = db_classic.labels_
                # D0 CommonNN:
                db_d0 = CommonNNClustering(eps=i, min_samples=minPTS, metric="precomputed").fit(datasets_dict["d0_distances"])
                db_d0_labels_pred = db_d0.labels_

                if len(set(db_classic_labels_pred)) >= 2 and len(set(db_d0_labels_pred)) >= 2:
                    # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

                    # Saving Silhouette score in a csv file:
                    writer.writerow([minPTS, i, metrics.silhouette_score(X, db_classic_labels_pred),
                                     metrics.silhouette_score(datasets_dict["d0_distances"], db_d0_labels_pred,
                                                              metric="precomputed")])

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

            # Plotting
            fig, ax = plt.subplots(1, 2)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

            ax[0].plot(
                datasets_dict["distances_interval"],
                db_classic_homogeneity_score,
                "r", label="classic")

            ax[0].plot(datasets_dict["distances_interval"],
                          db_d0_homogeneity_score,
                          "b--", label="d0-method")

            ax[0].legend(loc="upper right")
            ax[0].set_title("Homogeneity")
            ax[0].set_xlabel("epsilon distances")

            ax[1].plot(
                datasets_dict["distances_interval"],
                AMI_classic,
                "r", label="classic")
            ax[1].plot(
                datasets_dict["distances_interval"],
                AMI_d0,
                "b--", label="d0-method")


            ax[1].legend(loc="upper right")
            ax[1].set_title("AMI")
            ax[1].set_xlabel("epsilon distances")
            if customTicking:
                customTickingForYaxis(ax, [0, 1])

            fig.suptitle("{}-{}".format(method_name, datasets_dict["dataset_name"]))

            pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-CommonNN(Homogeneity,AMI)-default cost-minPTS{}"\
                .format(method_name, dataset_name, dataset_name, minPTS)
            plt.savefig(pathName)

            # Creating a figure that can be later changed
            with open(pathName + '.pkl', 'wb') as fid:
                pickle.dump(ax, fid)

            # *******************************************************
            fig, ax = plt.subplots(1, 2)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

            ax[0].plot(
                datasets_dict["distances_interval"],
                RAND_index_classic,
                "r", label="classic")
            ax[0].plot(
                datasets_dict["distances_interval"],
                RAND_index_d0,
                "b--", label="d0-method")

            ax[0].legend(loc="upper right")
            ax[0].set_title("Rand")
            ax[0].set_xlabel("epsilon distances")

            ax[1].plot(
                datasets_dict["distances_interval"],
                V_measure_classic,
                "r", label="Classic")
            ax[1].plot(
                datasets_dict["distances_interval"],
                V_measure_d0,
                "b--", label="d0-method")

            ax[1].legend(loc="upper right")
            ax[1].set_title("Vmeasure".format(datasets_dict["dataset_name"], method_name))
            ax[1].set_xlabel("epsilon distances")
            if customTicking:
                customTickingForYaxis(ax, [0, 1])
            fig.suptitle("{}-{}".format(method_name, datasets_dict["dataset_name"]))

            pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-CommonNN(Vmeasure,RAND)-default cost-minPTS{}"\
                .format(method_name, dataset_name, dataset_name, minPTS)
            plt.savefig(pathName)

            # Creating a figure that can be later changed
            with open(pathName + '.pkl', 'wb') as fid:
                pickle.dump(ax, fid)

            # F1 + AMI
            fig, ax = plt.subplots(1)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

            ax.plot(
                datasets_dict["distances_interval"],
                f1_classic,
                "r", label="classic")
            ax.plot(
                datasets_dict["distances_interval"],
                f1_d0,
                "b--", label="d0-method")

            ax.legend(loc="upper right")
            ax.set_title("F1".format(datasets_dict["dataset_name"], method_name))
            ax.set_xlabel("epsilon distances")
            if customTicking:
                customTickingForYaxis(ax, [0])

            pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-commonNN(F1)-default cost-minPTS{}" \
                .format(method_name, dataset_name, dataset_name, minPTS)
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


if __name__ == '__main__':
    # Method name:
    method_name = "CommonNN"
    # Enable/Disable customTicking on the Y-axis
    customTicking = True

    datasets = ["aggregation"]
    # datasets = ["aggregation", "breast_cancer", "coil", "D31", "diabetes",
    #             "flame", "genes", "iris", "isolet", "moons_1000",
    #             "olivetti", "pathbased", "phoneme", "R15", "spiral",
    #             "swiss_roll2D", "swiss_roll3D", "Umist", "wine", "digits"]

    evalMeasures(datasets, method_name, customTicking)