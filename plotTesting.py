import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import ticker


# Opening created pickle files for inspection

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")
fileNameToOpen = "Optics20figs"

with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"\
               f"OPTICS/Processed datasets/new_processed_figures/{fileNameToOpen}.pkl", 'rb') as fid:
    axes = pickle.load(fid)

    plt.show()


