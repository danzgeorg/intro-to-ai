import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# load only people with at least 60 photos (first run downloads ~200 MB)
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# show 15 faces with their names
fig, ax = plt.subplots(3, 5, figsize=(10,7))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()

# each face as one row of 2914 pixel values, and the person as a number 0-7
X = faces.data
y = faces.target

# fit PCA with all components to see how many we need
pca_all = PCA(svd_solver='randomized', random_state=42).fit(X)
plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

print("Variance kept by 150 components:", np.cumsum(
    pca_all.explained_variance_ratio_)[149])

# split before fitting PCA so the test faces stay completely unseen
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# randomised PCA down to 150 components learned from the training faces only
pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
pca.fit(X_train)

# apply the same projection to both sets
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

# SVM with an RBF kernel, trained on the 150 PCA features
model = SVC(kernel='rbf', gamma=0.001, C=5, class_weight='balanced')
model.fit(X_train_pca, y_train)

# predict the test faces (never seen by the PCA or the SVM) and score them
y_pred = model.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=faces.target_names))

# show 15 test faces with the predicted surname, red where its wrong
fig, ax = plt.subplots(3, 5, figsize=(10,7))
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1], 
                   color='black' if y_pred[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names, Incorrect Labels in Red', size=14)
plt.show()