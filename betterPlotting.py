import matplotlib.pyplot as plt
import pickle
from matplotlib import ticker

def colorTickLabelsAndPlot(row, column, axesNew, axesOld):
    axesNew[row][column].plot(axesOld.lines[0].get_xdata(), axesOld.lines[0].get_ydata(), "r--", label="classic")
    axesNew[row][column].plot(axesOld.lines[1].get_xdata(), axesOld.lines[1].get_ydata(), "b--", label="d0_method")

    ymax = max(axesNew[row][column].lines[0].get_ydata())
    ymax2 = max(axesNew[row][column].lines[1].get_ydata())

    if abs(ymax - ymax2) > 0.06: # so that the two ticks won't overlap
        axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
        flag = 0
        for t in axesNew[row][column].yaxis.get_ticklabels():
            if flag == 0:
                t.set_color("red")
                flag = 1
            else:
                t.set_color("blue")
    else:
        # there is overlap, therefore we show only the tick with the biggest value
        if ymax > ymax2:
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymax]))
            t = axesNew[row][column].yaxis.get_ticklabels()
            t[0].set_color("red")
        else:
            axesNew[row][column].yaxis.set_major_locator(ticker.FixedLocator([ymax2]))
            t = axesNew[row][column].yaxis.get_ticklabels()
            t[0].set_color("blue")

    axesNew[row][column].tick_params(axis='y', labelsize=7)
    axesNew[row][column].yaxis.set_label_coords(-0.25, 0.50)

def betterPlotting(saveToFile, pathName):
    row = 0
    fig, axesNew = plt.subplots(5, 5)
    for dataset in ["breast_cancer", "swiss_roll3D", "flame", "olivetti", "spiral"]:

        with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/new_Processed_Figures/"
                  f"{dataset}-commonNN-metrics.pkl", 'rb') as fid:
            axes = pickle.load(fid)
            fid.close()

        axes1 = axes[0].axes
        axes2 = axes[1].axes
        axes3 = axes[2].axes
        axes4 = axes[3].axes
        axes5 = axes[4].axes

        # Plot the data from the first subplot in the new figure
        colorTickLabelsAndPlot(row, 0, axesNew, axes1)

        axesNew[row][0].legend(loc="upper right")
        if row == 0:
            axesNew[row][0].set_title("F1")
        axesNew[row][0].set_ylabel(dataset)

        colorTickLabelsAndPlot(row, 1, axesNew, axes2)

        axesNew[row][1].legend(loc="upper right")
        if row == 0:
            axesNew[row][1].set_title("Homogeneity")

        colorTickLabelsAndPlot(row, 2, axesNew, axes3)

        axesNew[row][2].legend(loc="upper right")
        if row == 0:
            axesNew[row][2].set_title("AMI")

        colorTickLabelsAndPlot(row, 3, axesNew, axes4)

        axesNew[row][3].legend(loc="upper right")
        if row == 0:
           axesNew[row][3].set_title("Vmeasure")

        colorTickLabelsAndPlot(row, 4, axesNew, axes5)

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


# Creates a 5*5 subplot with results that are stored in previously created pickle files
def main():
    fileNameToStore = "CommonNN20figs"
    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/" \
               f"Processed datasets/new_Processed_Figures/{fileNameToStore}".format(fileNameToStore)

    saveToFile = True
    if saveToFile:
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







