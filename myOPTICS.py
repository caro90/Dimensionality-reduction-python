import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler



# #########################################################
# Compute OPTICS

X = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/coil.mat')
data = X.get('X')
labels = X.get('label')

euclidean_distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/coil_euclidean_distances.mat')
D = euclidean_distances.get('D')

distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/d0_distances sin method/coil_d0_distances.mat')
d0_distances = distances.get('d0_distances')
DMAX = distances.get('DMAX')
DMAX_avg = distances.get('DMAX_avg')
d_best = distances.get('d_best')

Dmax_temp_value = np.amax(D)

