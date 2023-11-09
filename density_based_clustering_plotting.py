import matplotlib.pyplot as plt
import pickle
from matplotlib import ticker


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

def customTickingForXaxis(ax, axesCounterList, d_best, epsilon_values): # check if needed
    if len(axesCounterList) > 1:
        d_best_index = epsilon_values.index(d_best)
        for axesCounter in axesCounterList:
            for index, x in enumerate(ax[axesCounter].xaxis.get_ticklabels()):
                if d_best_index == index:
                    x.set_color("red")
            xmin = min(ax[axesCounter].lines[0].get_xdata())
            xmax = max(ax[axesCounter].lines[1].get_xdata())
            ax[axesCounter].xaxis.set_major_locator(ticker.FixedLocator([d_best.item(), xmax]))

def plotting_figures_DBSAN_CommonNN(epsilon_values, datasets_dict, db_classic_homogeneity_score, db_d0_homogeneity_score,
                                    AMI_classic, AMI_d0, customTicking, method_name, dataset_name,
                                    RAND_index_classic, RAND_index_d0, V_measure_classic, f1_classic, f1_d0,
                                    V_measure_d0, j):
    # Plotting
    fig, ax = plt.subplots(1, 2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        epsilon_values,
        db_classic_homogeneity_score,
        "r", label="classic")

    ax[0].plot(epsilon_values,
               db_d0_homogeneity_score,
               "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Homogeneity")
    ax[0].set_xlabel("epsilon distances")

    ax[1].plot(
        epsilon_values,
        AMI_classic,
        "r", label="classic")
    ax[1].plot(
        epsilon_values,
        AMI_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("AMI")
    ax[1].set_xlabel("epsilon distances")

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-DBSCAN(Homogeneity,AMI)-default cost-Min_pts {}" \
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    # *******************************************************
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        epsilon_values,
        RAND_index_classic,
        "r", label="classic")
    ax[0].plot(
        epsilon_values,
        RAND_index_d0,
        "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Rand")
    ax[0].set_xlabel("epsilon distances")

    ax[1].plot(
        epsilon_values,
        V_measure_classic,
        "r", label="Classic")
    ax[1].plot(
        epsilon_values,
        V_measure_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ")
    ax[1].set_xlabel("epsilon distances")

    if customTicking:
        customTickingForYaxis(ax, [0, 1])  # *****************
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-DBSCAN(Vmeasure,RAND)-default cost-Min_pts {}" \
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1)
    plt.plot(
        epsilon_values,
        f1_classic,
        "r", label="classic")
    plt.plot(
        epsilon_values,
        f1_d0,
        "b--", label="d0-method")

    plt.legend(loc="upper right")
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))
    plt.title("F1 score")

    if customTicking:
        customTickingForYaxis(ax, [0])

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-DBSCAN(f1)-default cost-Min_pts {}" \
        .format(method_name, dataset_name, dataset_name, j)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)


def plotting_figures_OPTICS(datasets_dict, db_classic_homogeneity_score,
                        db_d0_homogeneity_score,
                        AMI_classic, AMI_d0, customTicking, method_name, dataset_name,
                        RAND_index_classic, RAND_index_d0, V_measure_classic, f1_classic,
                        f1_d0, V_measure_d0, min_pts_list):
    # Plotting
    fig, ax = plt.subplots(1, 2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        min_pts_list,
        db_classic_homogeneity_score,
        "r--", label="classic")

    ax[0].plot(min_pts_list,
               db_d0_homogeneity_score,
               "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Homogeneity")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        AMI_classic,
        "r--", label="classic")
    ax[1].plot(
        min_pts_list,
        AMI_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("AMI")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-OPTICS(Homogeneity,AMI)-default cost" \
        .format(method_name, dataset_name, dataset_name)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        min_pts_list,
        RAND_index_classic,
        "r--", label="classic")
    ax[0].plot(
        min_pts_list,
        RAND_index_d0,
        "b--", label="d0-method")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Rand")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        V_measure_classic,
        "r--", label="Classic")
    ax[1].plot(
        min_pts_list,
        V_measure_d0,
        "b--", label="d0-method")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-OPTICS(Vmeasure,RAND)-default cost" \
        .format(method_name, dataset_name, dataset_name)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1)
    plt.plot(
        min_pts_list,
        f1_classic,
        "r--", label="classic")
    plt.plot(
        min_pts_list,
        f1_d0,
        "b--", label="d0-method")

    plt.legend(loc="upper right")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))
    plt.title("F1 score")

    if customTicking:
        customTickingForYaxis(ax, [0])

    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.4/default cost function/{}/{}/{}-OPTICS(f1)-default cost" \
        .format(method_name, dataset_name, dataset_name)
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
