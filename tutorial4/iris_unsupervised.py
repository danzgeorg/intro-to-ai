import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

# load the data
iris = load_iris()
X = iris.data
y = iris.target

# PCA from 4 dimensions down to 2
pca = PCA(n_components=2)
pca.fit(X)
X_2d = pca.transform(X) # each flower is now 2 numbers

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total kept:", pca.explained_variance_ratio_.sum())

# plot the projected data, coloured by the true species (labels used for colour)
for k, name in enumerate(iris.target_names):
    plt.scatter(X_2d[y == k, 0], X_2d[y == k, 1], label=name, s=20)

plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.title("Iris projected onto 2 principal components")
plt.legend()
plt.show()

# cllusters are numbered 0,1,2 in an arbitrary order so give each cluster the
# most common true label inside it ("mask")
def match_labels(clusters, y_true, n_clusters):
    labels = np.zeros_like(clusters)
    for i in range(n_clusters):
        mask = (clusters == i)
        labels[mask] = mode(y_true[mask])[0]
    return labels

# k-means with 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
kmeans.fit(X)
clusters_km = kmeans.predict(X)
labels_km = match_labels(clusters_km, y, 3)

print("\nk-means accuracy:", accuracy_score(y, labels_km))
print(confusion_matrix(y, labels_km))

# Gaussian mixture model with 3 components
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)
clusters_gmm = gmm.predict(X)
labels_gmm = match_labels(clusters_gmm, y, 3)

print("\nGMM accuracy:", accuracy_score(y, labels_gmm))
print(confusion_matrix(y, labels_gmm))

# show both clustering side by side on the PCA plot
fig, ax = plt.subplots(1, 2, figsize= (11, 4))
ax[0].scatter(X_2d[:,0], X_2d[:, 1], c=labels_km, cmap='viridis', s=20)
ax[0].set_title("k-means clusters")
ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_gmm, cmap='viridis', s=20)
ax[1].set_title("GMM clusters")
plt.show()