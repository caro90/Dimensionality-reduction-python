import matplotlib.pyplot as plt
import pickle
from matplotlib import ticker
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator


def format_ticks(x, pos):
    # Define a custom tick formatter function
    return f'{x:.2f}'


def color_tick_labels_and_plot_euclid_piecewiselinear(row, column, axesNew, axesOld):
    axesNew[row][column].plot(axesOld.lines[0].get_xdata(), axesOld.lines[0].get_ydata(), "r--", label="classic")
    axesNew[row][column].plot(axesOld.lines[1].get_xdata(), axesOld.lines[1].get_ydata(), "b--", label="d0_method")

    ymax = max(axesNew[row][column].lines[0].get_ydata())
    ymax2 = max(axesNew[row][column].lines[1].get_ydata())
    ymin = 0
    # adjusting this value manually to avoid overlapping y-axis values:
    yaxis_pad = 0.01

    if abs(ymax - ymax2) > yaxis_pad:
        # so that the two ticks won't overlap
        if (abs(ymax - ymin) > yaxis_pad and abs(ymax2 - ymin) > yaxis_pad):
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymin, ymax, ymax2]))
            flag = 0
            for t1 in axesNew[row][column].yaxis.get_ticklabels():
                if flag == 0:
                    flag = 1
                elif flag == 1:
                    t1.set_color("red")
                    flag = 2
                elif flag == 2:
                    t1.set_color("blue")
        else:
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
            flag = 0
            for t2 in axesNew[row][column].yaxis.get_ticklabels():
                if flag == 0:
                    t2.set_color("red")
                    flag = 1
                elif flag == 1:
                    t2.set_color("blue")
    else:
        # there is overlap, therefore we show only the tick with the biggest value
        if ymax > ymax2:
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymin, ymax]))
            t3 = axesNew[row][column].yaxis.get_ticklabels()
            t3[1].set_color("red")
        else:
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymin, ymax2]))
            t4 = axesNew[row][column].yaxis.get_ticklabels()
            t4[1].set_color("blue")


    axesNew[row][column].tick_params(axis='y', labelsize=7)
    axesNew[row][column].yaxis.set_label_coords(-0.12, 0.50)
    axesNew[row][column].yaxis.set_major_formatter(
        '{:.2f}'.format)  # reducing the number of decimal points that are being displayed
    axesNew[row][column].set_ylim(0)


def color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, column, axesNew, axesOld):
    axesNew[row][column].plot(axesOld.lines[0].get_xdata(), axesOld.lines[0].get_ydata(),
                              linestyle=axesOld.lines[0].get_linestyle(), label="Euclidean")
    axesNew[row][column].plot(axesOld.lines[1].get_xdata(), axesOld.lines[1].get_ydata(),
                              linestyle=axesOld.lines[1].get_linestyle(), label="Piecewise linear d0")
    axesNew[row][column].plot(axesOld.lines[2].get_xdata(), axesOld.lines[2].get_ydata(),
                              linestyle=axesOld.lines[2].get_linestyle(), label="Exponential d0")

    axesNew[row][column].yaxis.set_major_formatter(axesOld.axes.yaxis.get_major_formatter())
    axesNew[row][column].yaxis.set_major_locator(axesOld.axes.yaxis.get_major_locator())

    # Set custom tick formatter for y-axis
    axesNew[row][column].yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    # Transfer tick colors from previous axes
    for tick_label in axesOld.axes.yaxis.get_ticklabels():
        axesNew[row][column].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())

    # Transfer line colors
    for source_line, target_line in zip(axesOld.axes.lines, axesNew[row][column].lines):
        target_line.set_color(source_line.get_color())

    if row == 0 and column == 4:
        # Position of the legend
        axesNew[row][column].legend(loc='upper right', bbox_to_anchor=(1.60, 1.05))


def density_based_clustering_20figures_plotting(list_of_datasets, saveToFile, load_pickles_loc_path_folder, clustering_method, ylabel, filename):

    counter = 0
    for datasets in list_of_datasets:
        row = 0
        fig, axesNew = plt.subplots(5, 5)
        for dataset in datasets:
            counter += 1
            path_to_load_pickle = (f"{load_pickles_loc_path_folder}/{clustering_method}/{dataset}/Best/"
                                   f"{dataset}-{clustering_method}-metrics.pkl")
            if clustering_method == "OPTICS":
                path_to_load_pickle = (f"{load_pickles_loc_path_folder}/{clustering_method}/{dataset}/"
                                       f"{dataset}-{clustering_method}-metrics.pkl")
            with open(path_to_load_pickle, 'rb') as fid:
                loaded_axes = pickle.load(fid)
                fid.close()

            # Plot the data from the first subplot in the new figure
            color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, 0, axesNew, loaded_axes[0])
            if row == 0:
                axesNew[row][0].set_title("F1", fontsize=15)
            axesNew[row][0].set_ylabel(dataset, fontsize=15)

            color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, 1, axesNew, loaded_axes[1])
            # Use to avoid overlapping values on the x-axis
            if dataset == 'coil':
                axesNew[row][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
                axesNew[row][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
                axesNew[row][2].xaxis.set_major_locator(MaxNLocator(nbins=5))
                axesNew[row][3].xaxis.set_major_locator(MaxNLocator(nbins=5))
                axesNew[row][4].xaxis.set_major_locator(MaxNLocator(nbins=5))

            if row == 0:
                axesNew[row][1].set_title("Homogeneity", fontsize=15)
            color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, 2, axesNew, loaded_axes[2])

            if row == 0:
                axesNew[row][2].set_title("AMI", fontsize=15)
            color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, 3, axesNew, loaded_axes[3])

            if row == 0:
               axesNew[row][3].set_title("Vmeasure", fontsize=15)
            color_tick_labels_and_plot_euclid_piecewiselinear_exp(row, 4, axesNew, loaded_axes[4])

            if row == 0:
                axesNew[row][4].set_title("RAND", fontsize=15)
            row += 1

            if row == 4:
                axesNew[row][2].set_xlabel(f"{ylabel}", fontsize=15)

        if saveToFile:
            # Storing as a pickle file
            # Creating a figure that can be later changed
            with open(f"{load_pickles_loc_path_folder}/{filename}-{counter}.pkl", 'wb') as fid:
                pickle.dump(axesNew, fid)
                fid.close()


def density_based_clustering_5figures_plotting(datasets, path, method_name):
    # Creating a plot with 5 figures (Homo,Vmeasure,Rand,AMI,F1) for each dataset

    for dataset in datasets:
        dataset_name = dataset

        p = Path(f"{path}/{method_name}/{dataset_name}/Best/")
        if method_name == "OPTICS":
            p = Path(f"{path}/{method_name}/{dataset_name}/")

        # If the intermediate folders do not exist, then they are being created
        os.makedirs(p, exist_ok=True)

        matching_files_f1 = [filename for filename in p.glob('**/*.pkl') if 'f1' in str(filename)]
        with open(f'{p}/{matching_files_f1[0].name}', 'rb') as fid:
            fig_f1 = pickle.load(fid)
            fid.close()

        matching_files_homo_ami = [filename for filename in p.glob('**/*.pkl') if 'Homogeneity' in str(filename)]
        with open(f'{p}/{matching_files_homo_ami[0].name}', 'rb') as fid:
            fig_homo_ami = pickle.load(fid)
            fid.close()

        matching_files_vmeasure_rand = [filename for filename in p.glob('**/*.pkl') if 'Vmeasure' in str(filename)]
        with open(f'{p}/{matching_files_vmeasure_rand[0].name}', 'rb') as fid:
            fig_vmeasure_rand = pickle.load(fid)
            fid.close()

        fileToStore = f"{dataset_name}-{method_name}-metrics"

        # Create a new figure with two identical subplots
        fig2, ax = plt.subplots(1, 5)
        plt.subplots_adjust(wspace=0.30)
        # fig2.tight_layout(h_pad=5)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)
        # Plot the data from the first subplot in the new figure
        ax[0].plot(fig_f1.axes.lines[0].get_xdata(), fig_f1.axes.lines[0].get_ydata(), "r--", label="cl")
        ax[0].plot(fig_f1.axes.lines[1].get_xdata(), fig_f1.axes.lines[1].get_ydata(), "b--", label="d0")
        ax[0].plot(fig_f1.axes.lines[2].get_xdata(), fig_f1.axes.lines[2].get_ydata(), "g--", label="exp-d0")

        # Transfer ticks
        ax[0].yaxis.set_major_formatter(fig_f1.axes.yaxis.get_major_formatter())
        ax[0].yaxis.set_major_locator(fig_f1.axes.yaxis.get_major_locator())
        # Transfer tick colors
        for tick_label in fig_f1.axes.yaxis.get_ticklabels():
            ax[0].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())
        ax[0].set_title("F1")
        ax[0].set_ylabel("Score")

        ax[1].plot(fig_homo_ami[0].axes.lines[0].get_xdata(), fig_homo_ami[0].axes.lines[0].get_ydata(), "r--", label="cl")
        ax[1].plot(fig_homo_ami[0].axes.lines[1].get_xdata(), fig_homo_ami[0].axes.lines[1].get_ydata(), "b--", label="d0")
        ax[1].plot(fig_homo_ami[0].axes.lines[2].get_xdata(), fig_homo_ami[0].axes.lines[2].get_ydata(), "g--", label="exp-d0")

        # Transfer ticks
        ax[1].yaxis.set_major_formatter(fig_homo_ami[0].axes.yaxis.get_major_formatter())
        ax[1].yaxis.set_major_locator(fig_homo_ami[0].axes.yaxis.get_major_locator())
        # Transfer tick colors
        for tick_label in fig_homo_ami[0].axes.yaxis.get_ticklabels():
            ax[1].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())
        ax[1].set_title("Homogeneity")

        ax[2].plot(fig_homo_ami[1].axes.lines[0].get_xdata(), fig_homo_ami[1].axes.lines[0].get_ydata(), "r--", label="cl")
        ax[2].plot(fig_homo_ami[1].axes.lines[1].get_xdata(), fig_homo_ami[1].axes.lines[1].get_ydata(), "b--", label="d0")
        ax[2].plot(fig_homo_ami[1].axes.lines[2].get_xdata(), fig_homo_ami[1].axes.lines[2].get_ydata(), "g--", label="exp-d0")

        # Transfer ticks
        ax[2].yaxis.set_major_formatter(fig_homo_ami[1].axes.yaxis.get_major_formatter())
        ax[2].yaxis.set_major_locator(fig_homo_ami[1].axes.yaxis.get_major_locator())
        # Transfer tick colors
        for tick_label in fig_homo_ami[1].axes.yaxis.get_ticklabels():
            ax[2].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())
        ax[2].set_title("AMI")
        ax[2].set_xlabel("epsilon distances")

        ax[3].plot(fig_vmeasure_rand[0].axes.lines[0].get_xdata(), fig_vmeasure_rand[0].axes.lines[0].get_ydata(), "r--", label="cl")
        ax[3].plot(fig_vmeasure_rand[0].axes.lines[1].get_xdata(), fig_vmeasure_rand[0].axes.lines[1].get_ydata(), "b--", label="d0")
        ax[3].plot(fig_vmeasure_rand[0].axes.lines[2].get_xdata(), fig_vmeasure_rand[0].axes.lines[2].get_ydata(), "g--", label="exp-d0")

        # Transfer ticks
        ax[3].yaxis.set_major_formatter(fig_vmeasure_rand[0].axes.yaxis.get_major_formatter())
        ax[3].yaxis.set_major_locator(fig_vmeasure_rand[0].axes.yaxis.get_major_locator())
        # Transfer tick colors
        for tick_label in fig_vmeasure_rand[0].axes.yaxis.get_ticklabels():
            ax[3].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())
        ax[3].set_title("Vmeasure")

        ax[4].plot(fig_vmeasure_rand[1].axes.lines[0].get_xdata(), fig_vmeasure_rand[1].axes.lines[0].get_ydata(),
                   linestyle=fig_vmeasure_rand[1].axes.lines[0].get_linestyle(),
                   label=fig_vmeasure_rand[1].axes.lines[0].get_label(),
                   color=fig_vmeasure_rand[1].axes.lines[0].get_color())
        ax[4].plot(fig_vmeasure_rand[1].axes.lines[1].get_xdata(), fig_vmeasure_rand[1].axes.lines[1].get_ydata(),
                   linestyle=fig_vmeasure_rand[1].axes.lines[1].get_linestyle(),
                   label=fig_vmeasure_rand[1].axes.lines[1].get_label(),
                   color=fig_vmeasure_rand[1].axes.lines[1].get_color())
        ax[4].plot(fig_vmeasure_rand[1].axes.lines[2].get_xdata(), fig_vmeasure_rand[1].axes.lines[2].get_ydata(),
                   linestyle=fig_vmeasure_rand[1].axes.lines[2].get_linestyle(),
                   label=fig_vmeasure_rand[1].axes.lines[2].get_label(),
                   color=fig_vmeasure_rand[1].axes.lines[2].get_color())

        # Transfer ticks
        ax[4].yaxis.set_major_formatter(fig_vmeasure_rand[1].axes.yaxis.get_major_formatter())
        ax[4].yaxis.set_major_locator(fig_vmeasure_rand[1].axes.yaxis.get_major_locator())
        # Transfer tick colors
        for tick_label in fig_vmeasure_rand[1].axes.yaxis.get_ticklabels():
            ax[4].yaxis.get_major_ticks()[0].label1.set_color(tick_label.get_color())

        # Position of the legend
        ax[4].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax[4].set_title("Vmeasure")

        # Storing as a pickle file
        path_name = f"{p}/{fileToStore}"
        # Creating a figure that can be later changed
        with open(path_name + '.pkl', 'wb') as fid:
            pickle.dump(ax, fid)

        plt.close('all')


def main():

    # Configuration settings:

    # Method name can be:
    # DBSCAN or CommonNN or OPTICS
    clustering_method = "OPTICS"
    version = "Version 0.5"
    sampling_function = "d0_distances_sin_method"
    cost_function = "default cost+cost 4"

    datasets = ["aggregation", "breast_cancer", "coil", "D31", "diabetes",
                "digits", "flame", "genes", "iris", "isolet",
                "moons_1000", "olivetti", "pathbased", "phoneme", "R15",
                "spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]

    load_pickles_loc_path_folder = (f"/home/arch/PycharmProjects/Dimensionality reduction results/{version}/"
                                    f"{sampling_function}/{cost_function}/20 percent dijkstra points - 100 percent backtracking points")

    openFile = True
    saveToFile = True

    # Creates a 1*5 subplot with results that are stored in previously created pickle files:
    density_based_clustering_5figures_plotting(datasets, load_pickles_loc_path_folder, clustering_method)

    # Creating a 5*5 subplot:
    ylabel = "MinPts"
    fileNameToStore = f"{clustering_method}20figs-result"

    datasets = [["isolet", "coil", "digits", "genes", "phoneme"],
                ["aggregation", "breast_cancer", "D31", "diabetes", "flame"],
                ["iris", "moons_1000", "olivetti", "pathbased", "R15"],
                ["spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]]


    if saveToFile:
        density_based_clustering_20figures_plotting(datasets, saveToFile, load_pickles_loc_path_folder,
                                                    clustering_method, ylabel, fileNameToStore)

    # For debugging purposes
    if openFile:
        plt.close('all')
        with open(f"{load_pickles_loc_path_folder}/{fileNameToStore}-5.pkl", 'rb') as fid:
            axes = pickle.load(fid)
        with open(f"{load_pickles_loc_path_folder}/{fileNameToStore}-10.pkl", 'rb') as fid:
            axes = pickle.load(fid)
        with open(f"{load_pickles_loc_path_folder}/{fileNameToStore}-15.pkl", 'rb') as fid:
            axes = pickle.load(fid)
        with open(f"{load_pickles_loc_path_folder}/{fileNameToStore}-20.pkl", 'rb') as fid:
            axes = pickle.load(fid)
        plt.show()

if __name__ == '__main__':
    main()







