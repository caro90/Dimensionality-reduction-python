import scipy.io
import numpy as np

def load_datasets(dataset_name):
    # Load datasets:
    X = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/{}.mat'.format(dataset_name))
    data = X.get('X')
    labels = X.get('labels')

    euclidean_distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/{}_euclidean_distances.mat'.format(dataset_name))
    D = euclidean_distances.get('D')

    distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/Version 0.3/d0_distances_sin_method/cost 1 default/lambda 10000/{}_d0_distances.mat'.format(dataset_name))
    d0_distances = distances.get('d0_distances')
    DMAX = distances.get('DMAX')
    DMAX_avg = distances.get('DMAX_avg')
    d_best = distances.get('d_best')

    Dmax_temp_value = np.amax(D)
    # In matlab: T=D+eye(size(D)).*Dmax_temp_value;
    T = D + np.diag(np.full(D.shape[1],1)) * Dmax_temp_value
    Dmin_temp_value = np.amin(T)
    labels = np.reshape(labels, D.shape[1])

    distances_interval = np.linspace(Dmin_temp_value, Dmax_temp_value, 40)
    if distances_interval[0] == 0:
        distances_interval = np.delete(distances_interval, 0)

    return {'dataset_name': dataset_name, "data": data, "D": D, "distances": distances, "d0_distances": d0_distances,
            "DMAX": DMAX, "DMAX_avg": DMAX_avg, "d_best": d_best, "Dmax_temp_value": Dmax_temp_value,
            "Dmin_temp_value": Dmin_temp_value, "labels": labels, "distances_interval": distances_interval}


def load_datasets_new_costs(dataset_name):
    distances = scipy.io.loadmat('/home/arch/Matlab/Dimensionality Reduction/mat_files/Version 0.2/d0_distances sin method/cost 4/{}_d0_distances.mat'.format(dataset_name))
    d0_distances = distances.get('d0_distances')

    return {"d0_distances": d0_distances}
