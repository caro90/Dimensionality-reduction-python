import os

# Opening created pickle files for inspection

mylist = os.listdir("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN")
fileNameToOpen = "Optics20figs"
fileNameToOpen = "aggregation-DBSCAN(f1)-default cost-Min_pts 2.pkl"
dataset = "aggregation"


with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"\
               f"OPTICS/Processed datasets/new_processed_figures/{fileNameToOpen}.pkl", 'rb') as fid:
               f"DBSCAN/Extended/{dataset}/{fileNameToOpen}", 'rb') as fid:
    axes = pickle.load(fid)

    plt.show()