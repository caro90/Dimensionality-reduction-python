import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import ticker

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")
for dataset in mylist:

    with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"
              f"OPTICS/Processed datasets/new_processed_figures/{dataset}-OPTICS-metrics.pkl", 'rb') as fid:
        axes = pickle.load(fid)

    plt.suptitle(f'{dataset}')


    for subplot in range(0,4):
        ymax = max(axes[subplot].lines[0].get_ydata())
        ymax2 = max(axes[subplot].lines[1].get_ydata())

        if abs(ymax-ymax2) > 0.04:
            continue

        if ymax > ymax2:
            axes[subplot].yaxis.set_major_locator(ticker.FixedLocator([ymax]))
            t = axes[subplot].yaxis.get_ticklabels()
            t[0].set_color("red")
        else:
            axes[subplot].yaxis.set_major_locator(ticker.FixedLocator([ymax2]))
            t = axes[subplot].yaxis.get_ticklabels()
            t[0].set_color("blue")

    ax = plt.gca()
    ax.figure.set_size_inches(30, 5)
    plt.show()


