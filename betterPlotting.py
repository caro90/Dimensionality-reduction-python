import matplotlib.pyplot as plt
import pickle
from matplotlib import ticker

def colorTickLabelsAndPlot(row, column, axesNew, axesOld):
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

def betterPlotting(saveToFile, pathName, clustering_method, ylabel):
    row = 0
    fig, axesNew = plt.subplots(5, 5)
    """ 
    OPTICS datasets pick up order:
    ["isolet", "coil", "digits", "genes", "phoneme"]
    ["aggregation", "breast_cancer", "D31", "diabetes", "flame"]
    ["iris", "moons_1000", "olivetti", "pathbased", "R15"]
    ["spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]
    """

    """ 
    CommonNN datasets pick up order:
    ["isolet", "coil", "digits", "genes", "phoneme"]
    ["aggregation", "breast_cancer", "D31", "diabetes", "flame"]
    ["iris", "moons_1000", "olivetti", "pathbased", "R15"]
    ["spiral", "swiss_roll2D", "swiss_roll3D", "Umist", "wine"]
    """

    for dataset in ["isolet", "coil", "digits", "genes", "phoneme"]:

        with open(f"/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/{clustering_method}/Processed datasets/new_processed_figures/"
                  f"{dataset}-{clustering_method}-metrics.pkl", 'rb') as fid:
            axes = pickle.load(fid)
            fid.close()

        axes1 = axes[0].axes
        axes2 = axes[1].axes
        axes3 = axes[2].axes
        axes4 = axes[3].axes
        axes5 = axes[4].axes

        # Plot the data from the first subplot in the new figure
        colorTickLabelsAndPlot(row, 0, axesNew, axes1)

        if row == 0:
            axesNew[row][0].set_title("F1", fontsize= 15)
        axesNew[row][0].set_ylabel(dataset, fontsize=15)
        axesNew[row][0].lines[0].set_linestyle("solid")
        axesNew[row][0].legend(loc="upper right")

        colorTickLabelsAndPlot(row, 1, axesNew, axes2)

        # Use to avoid overlapping values on the x-axis
        # if dataset == 'coil':
        #     axesNew[row][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
        #     axesNew[row][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
        #     axesNew[row][2].xaxis.set_major_locator(MaxNLocator(nbins=5))
        #     axesNew[row][3].xaxis.set_major_locator(MaxNLocator(nbins=5))
        #     axesNew[row][4].xaxis.set_major_locator(MaxNLocator(nbins=5))
        #-------
        axesNew[row][1].legend(loc="upper right")
        if row == 0:
            axesNew[row][1].set_title("Homogeneity", fontsize= 15)
        axesNew[row][1].lines[0].set_linestyle("solid")
        colorTickLabelsAndPlot(row, 2, axesNew, axes3)
        axesNew[row][2].legend(loc="upper right")


        if row == 0:
            axesNew[row][2].set_title("AMI", fontsize= 15)
        axesNew[row][2].lines[0].set_linestyle("solid")

        colorTickLabelsAndPlot(row, 3, axesNew, axes4)
        axesNew[row][3].legend(loc="upper right")
        axesNew[row][1].legend(loc="upper right")

        if row == 0:
           axesNew[row][3].set_title("Vmeasure", fontsize= 15)
        axesNew[row][3].lines[0].set_linestyle("solid")

        colorTickLabelsAndPlot(row, 4, axesNew, axes5)

        if row == 0:
            axesNew[row][4].set_title("RAND", fontsize= 15)
        axesNew[row][4].lines[0].set_linestyle("solid")
        axesNew[row][4].legend(loc="upper right")
        row += 1

        if row == 4:
            axesNew[row][2].set_xlabel(f"{ylabel}", fontsize= 15)
    if saveToFile:
        # Storing as a pickle file
        # Creating a figure that can be later changed
        with open(pathName + '.pkl', 'wb') as fid:
            pickle.dump(axesNew, fid)
            fid.close()
    plt.close(fig)

# Creates a 5*5 subplot with results that are stored in previously created pickle files
def main():

    clustering_method = "OPTICS"
    ylabel = "Epsilon distances"
    fileNameToStore = f"{clustering_method}20figs-result1-new"
    pathName = f"/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/{clustering_method}/" \
               f"Processed datasets/new_processed_figures/{fileNameToStore}"

    saveToFile = True
    if saveToFile:
        betterPlotting(saveToFile, pathName, clustering_method, ylabel)

    openFile = True
    if openFile:
        with open(pathName+".pkl", 'rb') as fid:
            axes = pickle.load(fid)
        plt.show()

if __name__ == '__main__':
    main()







