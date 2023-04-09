import matplotlib.pyplot as plt
import pickle
import os

def betterPlotting(saveToFile, pathName):
    mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")
    row = 0
    fig, axesNew = plt.subplots(5, 5)
    for dataset in ["breast_cancer", "coil", "digits", "genes", "isolet"]:

        with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"
                  f"OPTICS/Processed datasets/new_processed_figures/{dataset}-OPTICS-metrics.pkl", 'rb') as fid:
            axes = pickle.load(fid)
            fid.close()

        axes1 = axes[0].axes
        axes2 = axes[1].axes
        axes3 = axes[2].axes
        axes4 = axes[3].axes
        axes5 = axes[4].axes

        # Plot the data from the first subplot in the new figure
        axesNew[row][0].plot(axes1.lines[0].get_xdata(), axes1.lines[0].get_ydata(), "r--", label="classic")
        axesNew[row][0].plot(axes1.lines[1].get_xdata(), axes1.lines[1].get_ydata(), "b--", label="d0_method")

        axesNew[row][0].legend(loc="upper right")
        if row == 0:
            axesNew[row][0].set_title("F1")
        axesNew[row][0].set_ylabel(dataset)

        axesNew[row][1].plot(axes2.lines[0].get_xdata(), axes2.lines[0].get_ydata(), "r--", label="classic")
        axesNew[row][1].plot(axes2.lines[1].get_xdata(), axes2.lines[1].get_ydata(), "b--", label="d0_method")

        axesNew[row][1].legend(loc="upper right")
        if row == 0:
            axesNew[row][1].set_title("Homogeneity")

        axesNew[row][2].plot(axes3.lines[0].get_xdata(), axes3.lines[0].get_ydata(), "r--", label="classic")
        axesNew[row][2].plot(axes3.lines[1].get_xdata(), axes3.lines[1].get_ydata(), "b--", label="d0_method")

        axesNew[row][2].legend(loc="upper right")
        if row == 0:
            axesNew[row][2].set_title("AMI")

        axesNew[row][3].plot(axes4.lines[0].get_xdata(), axes4.lines[0].get_ydata(), "r--", label="classic")
        axesNew[row][3].plot(axes4.lines[1].get_xdata(), axes4.lines[1].get_ydata(), "b--", label="d0_method")

        axesNew[row][3].legend(loc="upper right")
        if row == 0:
           axesNew[row][3].set_title("Vmeasure")

        axesNew[row][4].plot(axes5.lines[0].get_xdata(), axes5.lines[0].get_ydata(), "r--", label="classic")
        axesNew[row][4].plot(axes5.lines[1].get_xdata(), axes5.lines[1].get_ydata(), "b--", label="d0_method")

        axesNew[row][4].legend(loc="upper right")
        if row == 0:
            axesNew[row][4].set_title("RAND")

        row += 1

    if saveToFile:
        # Storing as a pickle file
        # Creating a figure that can be later changed
        with open(pathName + '.pkl', 'wb') as fid:
            pickle.dump(axesNew, fid)
            fid.close()
    plt.close(fig)

def main():

    saveToFile = False
    fileNameToStore = "Optics20figs"
    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/" \
               "OPTICS/Processed datasets/new_processed_figures/{}" \
        .format(fileNameToStore)
    betterPlotting(saveToFile, pathName)

    openFile = True
    if openFile:
        with open(pathName+".pkl", 'rb') as fid:
            axes = pickle.load(fid)
        #ax = plt.gca()
        #ax.figure.set_size_inches(30, 5)
        plt.show()

if __name__ == '__main__':
    main()







