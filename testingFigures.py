import matplotlib.pyplot as plt
import pickle


with open('myplot.pkl', 'rb') as fid:
    ax = pickle.load(fid)
plt.show()