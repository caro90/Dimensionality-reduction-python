import scipy
from sklearn.manifold import Isomap, trustworthiness
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr

from loadDatasets import load_datasets

dataset_name = 'phoneme'
path_cost_function = f'/home/arch/My_Data/Datasets/Dimensionality Reduction/mat_files/Version 0.5/d0_distances_sin_method/default cost/20 percent dijkstra points - 100 percent backtracking points - lambda 10000/{dataset_name}_d0_distances.mat'

# Load original dataset and distance matrices
X = scipy.io.loadmat(f'/home/arch/My_Data/Datasets/Dimensionality Reduction/mat_files/{dataset_name}.mat')
data = X.get('X')
labels = X.get('labels')
euclidean_distances = scipy.io.loadmat(f'/home/arch/My_Data/Datasets/Dimensionality Reduction/mat_files/{dataset_name}_euclidean_distances.mat')
D = euclidean_distances.get('D')
distances = scipy.io.loadmat(path_cost_function)

# Fix: Increase the number of neighbors to ensure a connected graph
n_neighbors = 15  # Increase n_neighbors to avoid disconnected components

# Use Isomap with the precomputed distance matrix (d0)
isomap = Isomap(n_neighbors=n_neighbors, metric='precomputed')
X_isomap = isomap.fit_transform(distances.get('d0_distances'))

# Use Isomap with original data
isomap_original = Isomap(n_neighbors=n_neighbors)
X_isomap_original = isomap_original.fit_transform(data)

### Metrics Comparison
# 1. Trustworthiness comparison
trust_isomap = trustworthiness(data, X_isomap)
trust_isomap_original = trustworthiness(data, X_isomap_original)

print(f"Trustworthiness (Isomap d0): {trust_isomap}")
print(f"Trustworthiness (Isomap original): {trust_isomap_original}")

# 2. Mean Squared Error (MSE) between pairwise distances
# High-dimensional space (using geodesic distances from Isomap)
D_high_isomap = pairwise_distances(isomap.dist_matrix_)
D_low_isomap = pairwise_distances(X_isomap)

mse_isomap = np.mean((D_high_isomap - D_low_isomap) ** 2)
print("MSE between distances (Isomap d0):", mse_isomap)

# For original Isomap
D_high_original = pairwise_distances(data)  # Euclidean distances
D_low_original = pairwise_distances(X_isomap_original)

mse_isomap_original = np.mean((D_high_original - D_low_original) ** 2)
print("MSE between distances (Isomap original):", mse_isomap_original)

# 3. Procrustes analysis: Option 1 (on pairwise distance matrices)
# Perform Procrustes analysis on pairwise distance matrices (D_high and D_low)
mse_procrustes_isomap, _, _ = procrustes(D_high_isomap, D_low_isomap)
mse_procrustes_isomap_original, _, _ = procrustes(D_high_original, D_low_original)

print(f"Procrustes MSE (Isomap d0): {mse_procrustes_isomap}")
print(f"Procrustes MSE (Isomap original): {mse_procrustes_isomap_original}")

# 3. Procrustes analysis: Option 2 (on reduced dimensions)
# Note: You can also reduce the original data to match the dimensionality of X_isomap

# Example: Extract the first few dimensions from the original data to match the shape of X_isomap
data_reduced = data[:, :X_isomap.shape[1]]

# Perform Procrustes analysis between reduced original data and Isomap embedding
mse_procrustes_isomap_data, _, _ = procrustes(data_reduced, X_isomap)
mse_procrustes_isomap_original_data, _, _ = procrustes(data_reduced, X_isomap_original)

print(f"Procrustes MSE (Isomap d0, reduced): {mse_procrustes_isomap_data}")
print(f"Procrustes MSE (Isomap original, reduced): {mse_procrustes_isomap_original_data}")

# 4. Residual variance
D_high_flat = D_high_isomap.flatten()
D_low_flat = D_low_isomap.flatten()

correlation, _ = pearsonr(D_high_flat, D_low_flat)
residual_variance_isomap = 1 - correlation**2
print(f"Residual Variance (Isomap d0): {residual_variance_isomap}")

D_high_original_flat = D_high_original.flatten()
D_low_original_flat = D_low_original.flatten()

correlation_original, _ = pearsonr(D_high_original_flat, D_low_original_flat)
residual_variance_isomap_original = 1 - correlation_original**2
print(f"Residual Variance (Isomap original): {residual_variance_isomap_original}")

# 5. Spearman's rank correlation
spearman_corr_isomap, _ = spearmanr(D_high_flat, D_low_flat)
spearman_corr_isomap_original, _ = spearmanr(D_high_original_flat, D_low_original_flat)

print(f"Spearman's rank correlation (Isomap d0): {spearman_corr_isomap}")
print(f"Spearman's rank correlation (Isomap original): {spearman_corr_isomap_original}")
