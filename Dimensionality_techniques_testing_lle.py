import numpy as np
import scipy
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes
from scipy.stats import pearsonr

# Load dataset (same as before)
dataset_name = 'olivetti'
X = scipy.io.loadmat(f'/home/arch/My_Data/Datasets/Dimensionality Reduction/mat_files/{dataset_name}.mat')
data = X.get('X')
labels = X.get('labels')

# Load precomputed distances (replace with your actual path)
distances = scipy.io.loadmat(f'/home/arch/My_Data/Datasets/Dimensionality Reduction/mat_files/Version 0.5/'
                             f'd0_distances_sin_method/default cost/20 percent dijkstra points - 100 percent '
                             f'backtracking points - lambda 10000/{dataset_name}_d0_distances.mat')
precomputed_distances = distances.get('d0_distances')

# Step 1: Compute nearest neighbors using precomputed distances
nbrs = NearestNeighbors(n_neighbors=10, metric='precomputed').fit(precomputed_distances)
_, indices = nbrs.kneighbors(precomputed_distances)

# Step 2: Apply LLE with custom distance matrix (using precomputed neighbors)
lle_custom = LocallyLinearEmbedding(n_neighbors=10, method='standard')
X_lle_custom = lle_custom.fit_transform(precomputed_distances)

# Step 3: Standard LLE using the original data
lle_original = LocallyLinearEmbedding(n_neighbors=10)
X_lle_original = lle_original.fit_transform(data)

# Step 4: Comparison of metrics

# 1. Trustworthiness Comparison
trust_lle_custom = trustworthiness(data, X_lle_custom)
trust_lle_original = trustworthiness(data, X_lle_original)

print(f"Trustworthiness (LLE custom): {trust_lle_custom}")
print(f"Trustworthiness (LLE original): {trust_lle_original}")

# 2. Mean Squared Error (MSE) between pairwise distances
D_high = pairwise_distances(data)  # High-dimensional space

# For custom LLE
D_low_lle_custom = pairwise_distances(X_lle_custom)  # Low-dimensional space from custom LLE
mse_lle_custom = np.mean((D_high - D_low_lle_custom) ** 2)
print(f"Mean Squared Error (LLE custom): {mse_lle_custom}")

# For original LLE
D_low_lle_original = pairwise_distances(X_lle_original)  # Low-dimensional space from original LLE
mse_lle_original = np.mean((D_high - D_low_lle_original) ** 2)
print(f"Mean Squared Error (LLE original): {mse_lle_original}")

# 3. Procrustes analysis
mse_procrustes_lle_custom, _, _ = procrustes(data, X_lle_custom)
mse_procrustes_lle_original, _, _ = procrustes(data, X_lle_original)
print(f"Procrustes MSE (LLE custom): {mse_procrustes_lle_custom}")
print(f"Procrustes MSE (LLE original): {mse_procrustes_lle_original}")

# 4. Residual Variance
D_high_flat = D_high.flatten()

# Custom LLE residual variance
D_low_lle_custom_flat = D_low_lle_custom.flatten()
correlation_custom, _ = pearsonr(D_high_flat, D_low_lle_custom_flat)
residual_variance_lle_custom = 1 - correlation_custom**2
print(f"Residual Variance (LLE custom): {residual_variance_lle_custom}")

# Original LLE residual variance
D_low_lle_original_flat = D_low_lle_original.flatten()
correlation_original, _ = pearsonr(D_high_flat, D_low_lle_original_flat)
residual_variance_lle_original = 1 - correlation_original**2
print(f"Residual Variance (LLE original): {residual_variance_lle_original}")
