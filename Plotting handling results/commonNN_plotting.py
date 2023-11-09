import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as ticker
import os

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")
for dataset in mylist:
    datasetName = dataset

    with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/{}-commonNN(F1)-default cost.pkl".format(datasetName), 'rb') as fid:
        fig_f1 = pickle.load(fid)

    with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/{}-CommonNN(Homogeneity,AMI)-default cost.pkl".format(datasetName), 'rb') as fid:
        fig_homo_ami = pickle.load(fid)

    with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/{}-CommonNN(Vmeasure,RAND)-default cost.pkl".format(datasetName), 'rb') as fid:
        fig_vmeasure_rand = pickle.load(fid)

    fileToStore = "{}-commonNN-metrics".format(datasetName)

    # Get the first subplot from the saved figure
    ax1 = fig_f1.axes
    ax2 = fig_homo_ami[0].axes
    ax3 = fig_homo_ami[1].axes
    ax4 = fig_vmeasure_rand[0].axes
    ax5 = fig_vmeasure_rand[1].axes

    # Create a new figure with two identical subplots
    fig2, ax = plt.subplots(1, 5)
    #
    plt.subplots_adjust(wspace=0.30)

    #fig2.tight_layout(h_pad=5)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)
    # Plot the data from the first subplot in the new figure
    ax[0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), "r--", label="cl")
    ax[0].plot(ax1.lines[1].get_xdata(), ax1.lines[1].get_ydata(), "b--", label="d0")

    ax[0].legend(loc="upper right")
    ax[0].set_title("F1")
    ax[0].set_ylabel("Score")

    ymax = max(ax1.lines[0].get_ydata())
    ymax2 = max(ax1.lines[1].get_ydata())

    ax[0].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))

    flag = 0
    for t in ax[0].yaxis.get_ticklabels():
        if flag == 0:
            t.set_color("red")
            flag = 1
        else:
            t.set_color("blue")

    ax[1].plot(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata(), "r--", label="cl")
    ax[1].plot(ax2.lines[1].get_xdata(), ax2.lines[1].get_ydata(), "b--", label="d0")

    ax[1].legend(loc="upper right")
    ax[1].set_title("Homogeneity")

    ymax = max(ax2.lines[0].get_ydata())
    ymax2 = max(ax2.lines[1].get_ydata())

    ax[1].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
    flag = 0
    for t in ax[1].yaxis.get_ticklabels():
        if flag == 0:
            t.set_color("red")
            flag = 1
        else:
            t.set_color("blue")


    ax[2].plot(ax3.lines[0].get_xdata(), ax3.lines[0].get_ydata(), "r--", label="cl")
    ax[2].plot(ax3.lines[1].get_xdata(), ax3.lines[1].get_ydata(), "b--", label="d0")

    ax[2].legend(loc="upper right")
    ax[2].set_title("AMI")
    ax[2].set_xlabel("epsilon distances")

    ymax = max(ax3.lines[0].get_ydata())
    ymax2 = max(ax3.lines[1].get_ydata())

    ax[2].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
    flag = 0
    for t in ax[2].yaxis.get_ticklabels():
        if flag == 0:
            t.set_color("red")
            flag = 1
        else:
            t.set_color("blue")


    ax[3].plot(ax4.lines[0].get_xdata(), ax4.lines[0].get_ydata(), "r--", label="cl")
    ax[3].plot(ax4.lines[1].get_xdata(), ax4.lines[1].get_ydata(), "b--", label="d0")

    ax[3].legend(loc="upper right")
    ax[3].set_title("Vmeasure")

    ymax = max(ax4.lines[0].get_ydata())
    ymax2 = max(ax4.lines[1].get_ydata())

    ax[3].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
    flag = 0
    for t in ax[3].yaxis.get_ticklabels():
        if flag == 0:
            t.set_color("red")
            flag = 1
        else:
            t.set_color("blue")

    ax[4].plot(ax5.lines[0].get_xdata(), ax5.lines[0].get_ydata(), "r--", label="cl")
    ax[4].plot(ax5.lines[1].get_xdata(), ax5.lines[1].get_ydata(), "b--", label="d0")

    ax[4].legend(loc="upper right")
    ax[4].set_title("Vmeasure")

    ymax = max(ax5.lines[0].get_ydata())
    ymax2 = max(ax5.lines[1].get_ydata())

    ax[4].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
    flag = 0
    for t in ax[4].yaxis.get_ticklabels():
        if flag == 0:
            t.set_color("red")
            flag = 1
        else:
            t.set_color("blue")

    # Storing as a pickle file
    pathName = "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/newProcessedFigures/{}"\
            .format(fileToStore)
     # Creating a figure that can be later changed
    with open(pathName + '.pkl', 'wb') as fid:
            pickle.dump(ax, fid)

    #plt.show()
