import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as ticker


# datasetName = "olivetti"
# fileToStore = "{}-commonNN-metrics".format(datasetName)
with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/"
          "CommonNN/Processed datasets/newProcessedFigures/swiss_roll3D-commonNN-metrics.pkl", 'rb') as fid:
    fig = pickle.load(fid)

plt.show()


