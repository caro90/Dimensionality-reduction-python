import scipy.io
import numpy as np

def load_datasets():
    dataset_name = "coil"
    # Load datasets:
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

    # In matlab: T=D+eye(size(D)).*Dmax_temp_value;
    T = D + np.diag(np.full(D.shape[1],1)) * Dmax_temp_value
    Dmin_temp_value = np.amin(T)
    labels = np.reshape(labels, D.shape[1])

    distances_interval = np.linspace(Dmin_temp_value, Dmax_temp_value, 5)
    if distances_interval[0] == 0:
        distances_interval = np.delete(distances_interval, 0)

    return {'dataset_name': dataset_name, "data": data, "D": D, "distances": distances, "d0_distances": d0_distances,
            "DMAX": DMAX, "DMAX_avg": DMAX_avg, "d_best": d_best, "Dmax_temp_value": Dmax_temp_value,
            "Dmin_temp_value": Dmin_temp_value, "labels": labels, "distances_interval": distances_interval}
