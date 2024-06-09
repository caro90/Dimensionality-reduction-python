import matplotlib.pyplot as plt
import pickle
from matplotlib import ticker


def customTickingForYaxis(ax, axesCounterList):

    if len(axesCounterList) > 1:
        for axesCounter in axesCounterList:

            ymax1 = max(ax[axesCounter].lines[0].get_ydata())
            ymax2 = max(ax[axesCounter].lines[1].get_ydata())
            ymax3 = max(ax[axesCounter].lines[2].get_ydata())

            ax[axesCounter].yaxis.set_major_locator(ticker.FixedLocator([max(ymax1, ymax2, ymax3)]))
            ymax = max(ymax1, ymax2, ymax3)
            for finalYmax in ax[axesCounter].yaxis.get_ticklabels():
                if ymax == ymax1:
                    finalYmax.set_color("red")
                elif ymax == ymax2:
                    finalYmax.set_color("blue")
                else:
                    finalYmax.set_color("green")
    else:
        ymax1 = max(ax.lines[0].get_ydata())
        ymax2 = max(ax.lines[1].get_ydata())
        ymax3 = max(ax.lines[2].get_ydata())

        ax.yaxis.set_major_locator(ticker.FixedLocator([max(ymax1, ymax2, ymax3)]))
        ymax = max(ymax1, ymax2, ymax3)
        for ylabel in ax.yaxis.get_ticklabels():
            if ymax == ymax1:
                ylabel.set_color("red")
            elif ymax == ymax2:
                ylabel.set_color("blue")
            else:
                ylabel.set_color("green")


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

def plotting_figures_DBSAN_CommonNN(epsilon_values, datasets_dict,
                                    db_classic_homogeneity_score,
                                    db_d0_homogeneity_score,
                                    db_d0_alternative_homogeneity_score,
                                    AMI_classic, AMI_d0, AMI_d0_alternative,
                                    customTicking, method_name, dataset_name,
                                    RAND_index_classic,
                                    RAND_index_d0,
                                    RAND_index_d0_alternative,
                                    V_measure_classic, f1_classic, f1_d0, f1_d0_altenative,
                                    V_measure_d0,
                                    V_measure_d0_alternative, j, cost_function,
                                    save_loc_folder_path):
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

    ax[0].plot(epsilon_values,
               db_d0_alternative_homogeneity_score,
               "g--", label="d0-exponential")

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
    ax[1].plot(
        epsilon_values,
        AMI_d0_alternative,
        "g--", label="d0-exponential")

    ax[1].legend(loc="upper right")
    ax[1].set_title("AMI")
    ax[1].set_xlabel("epsilon distances")

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(Homogeneity,AMI)-{cost_function}-Min_pts {j}")
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
    ax[0].plot(
        epsilon_values,
        RAND_index_d0_alternative,
        "g--", label="d0-exponential")

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
    ax[1].plot(
        epsilon_values,
        V_measure_d0_alternative,
        "g--", label="d0-exponential")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ")
    ax[1].set_xlabel("epsilon distances")

    if customTicking:
        customTickingForYaxis(ax, [0, 1])
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(Vmeasure,RAND)-{cost_function}-Min_pts {j}")
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
    plt.plot(
        epsilon_values,
        f1_d0_altenative,
        "g--", label="d0-exponential")

    plt.legend(loc="upper right")
    fig.suptitle("{}-MinPts:{}-{}".format(method_name, j, datasets_dict["dataset_name"]))
    plt.title("F1 score")

    if customTicking:
        customTickingForYaxis(ax, [0])

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(f1)-{cost_function}-Min_pts {j}")
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)


def plotting_figures_OPTICS(datasets_dict, db_classic_homogeneity_score, db_d0_homogeneity_score,
                            db_d0_alternative_homogeneity_score,
                            ami_classic, ami_d0, ami_d0_alternative,
                            customTicking, method_name, dataset_name,
                            rand_index_classic,
                            rand_index_d0,
                            rand_index_d0_alternative, v_measure_classic,
                            f1_classic, f1_d0, f1_d0_alternative, v_measure_d0,
                            v_measure_d0_alternative, min_pts_list, cost_function,
                            save_loc_folder_path):

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

    ax[0].plot(min_pts_list,
               db_d0_alternative_homogeneity_score,
               "g--", label="d0-exponential")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Homogeneity")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        ami_classic,
        "r--", label="classic")
    ax[1].plot(
        min_pts_list,
        ami_d0,
        "b--", label="d0-method")

    ax[1].plot(
        min_pts_list,
        ami_d0_alternative,
        "g--", label="d0-exponential")

    ax[1].legend(loc="upper right")
    ax[1].set_title("AMI")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(Homogeneity,AMI)-{cost_function}")
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)

    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

    ax[0].plot(
        min_pts_list,
        rand_index_classic,
        "r--", label="classic")
    ax[0].plot(
        min_pts_list,
        rand_index_d0,
        "b--", label="d0-method")

    ax[0].plot(
        min_pts_list,
        rand_index_d0_alternative,
        "g--", label="d0-exponential")

    ax[0].legend(loc="upper right")
    ax[0].set_title("Rand")
    ax[0].set_xlabel("MinPts")

    ax[1].plot(
        min_pts_list,
        v_measure_classic,
        "r--", label="Classic")
    ax[1].plot(
        min_pts_list,
        v_measure_d0,
        "b--", label="d0-method")

    ax[1].plot(
        min_pts_list,
        v_measure_d0_alternative,
        "g--", label="d0-exponential")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Vmeasure ")
    ax[1].set_xlabel("MinPts")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))

    if customTicking:
        customTickingForYaxis(ax, [0, 1])

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(Vmeasure,RAND)-{cost_function}")
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
    plt.plot(
        min_pts_list,
        f1_d0_alternative,
        "g--", label="d0-exponential")

    plt.legend(loc="upper right")
    fig.suptitle("OPTICS-{}".format(datasets_dict["dataset_name"]))
    plt.title("F1 score")

    if customTicking:
        customTickingForYaxis(ax, [0])

    pathName = (f"{save_loc_folder_path}/"
                f"{dataset_name}-{method_name}(f1)-{cost_function}")
    plt.savefig(pathName)

    # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
