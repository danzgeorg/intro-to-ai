import numpy as np
from sklearn.datasets import load_breast_cancer

# load the data: 569 tumours, 30 numeric measurements each
data = load_breast_cancer()
X, y = data.data, data.target

print("X shape:", X.shape) # (rows, features)
print("Classes:", data.target_names) # what 0 and 1 mean
print("Counts:", np.bincount(y)) # how many of each class
print("First 5 features:", data.feature_names[:5])
print("Feature ranges (min, max) of the first 5:")
print(np.round(X[:, :5].min(axis=0), 3))
print(np.round(X[:, :5].max(axis=0), 3))

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

SEED = 42 # one fixed seed so every run (and every model) uses the same split

# 75% train, 25% test, stratify = y keeps the 37/63 class mix in both parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print("\nTrain:", X_train.shape, "Test:", X_test.shape)
print("Malignant share, train vs test:",
      round(np.mean(y_train==0), 3), round(np.mean(y_test==0), 3))

# always predict "benign" (1)
print("Always benign baseline:", round(np.mean(y_test==1), 3))

# first kNN: default settings (k=5), unscaled features
knn_raw = KNeighborsClassifier()
knn_raw.fit(X_train, y_train)
print("kNN, raw features, test accuracy:", 
      round(accuracy_score(y_test, knn_raw.predict(X_test)), 3))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# scale every feature to mean 0, standard deviation 1, then run kNN
knn_scaled = Pipeline([("scale", StandardScaler()),
                       ("knn", KNeighborsClassifier())])

# 5 fold CV on the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

raw_cv = cross_val_score(knn_raw, X_train, y_train, cv=cv)
scaled_cv = cross_val_score(knn_scaled, X_train, y_train, cv=cv)
print("\nkNN 5 fold CV accuracy, raw: ", np.round(raw_cv, 3), "mean", round(raw_cv.mean(), 3))
print("\nkNN 5 fold CV accuracy, scaled: ", np.round(scaled_cv, 3), "mean", round(scaled_cv.mean(), 3))

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# accuracy vs k: training score vs cross validated score
ks = range(1, 31)
train_acc, cv_acc = [], []
for k in ks:
    model = Pipeline([("scale", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=k))])
    cv_acc.append(cross_val_score(model, X_train, y_train, cv=cv).mean())
    train_acc.append(model.fit(X_train, y_train).score(X_train, y_train))
    
plt.figure(figsize=(7,4))
plt.plot(ks, train_acc, label="training")
plt.plot(ks, cv_acc, label="5 fold CV")
plt.xlabel("k (n_neighbours)")
plt.ylabel("accuracy")
plt.title("kNN: accuracy vs k")
plt.legend()
plt.show()

# grid search over k, weighting and distance type
# inside a Pipeline, parameter names are "stepname__parameter" (2 underscores)
knn_grid = {"knn__n_neighbors": list(range(1, 31, 2)),  # odd k avoids tied votes
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2]}                           # 1 = Manhattan, 2 = Euclidean

knn_search = GridSearchCV(knn_scaled, knn_grid, cv=cv, scoring="accuracy")
knn_search.fit(X_train, y_train)
print("\nkNN best settings:", knn_search.best_params_)
print("kNN best CV accuracy:", round(knn_search.best_score_, 3))

from sklearn.ensemble import RandomForestClassifier

# CV accuracy vs number of trees: look for where the curve goes flat
trees = [1,2,5,10,20,50,100,200,400]
rf_cv = [cross_val_score(RandomForestClassifier(n_estimators=t, random_state=SEED, n_jobs=-1),
                         X_train, y_train, cv=cv).mean() for t in trees]

plt.figure(figsize=(7,4))
plt.plot(trees, rf_cv, marker="o")
plt.xscale("log") # log scale spreads out in 1,2,5 ... 400 evenly
plt.xlabel("number of trees (log scale)")
plt.ylabel("5 fold CV accuracy")
plt.title("Random forest: accuracy vs number of trees")
plt.show()

# grid search: trees, features per split, max depth
rf = RandomForestClassifier(random_state=SEED, n_jobs=1)
rf_grid = {"n_estimators": [50, 100, 200, 400],
           "max_features": ["sqrt", "log2", None],  # None = all 30 features = plain bagging
           "max_depth": [None, 5, 10]}              # None = grow each tree fully

rf_search = GridSearchCV(rf, rf_grid, cv=cv, scoring="accuracy", n_jobs=1)
rf_search.fit(X_train, y_train)
print("\nRF best settings:", rf_search.best_params_)
print("RF best CV accuracy:", round(rf_search.best_score_, 3))

from sklearn.naive_bayes import GaussianNB

# gaussian NB: one normal distribution per feature per class, features assumed independent
gnb = GaussianNB()
gnb_grid = {"var_smoothing": np.logspace(-12,-3,10)} # 1e-12, 1e-11, ... 1e-3
gnb_search = GridSearchCV(gnb, gnb_grid, cv=cv, scoring="accuracy")
gnb_search.fit(X_train, y_train)
print("\nGNB best settings:", gnb_search.best_params_)
print("GNB best CV accuracy:", round(gnb_search.best_score_, 3))

import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

searches = {"kNN": knn_search, "Random Forest": rf_search, "Gaussian NB": gnb_search}
rows = []
for name, search in searches.items():
    y_pred = search.predict(X_test) # uses the best model found by the grid search
    cm = confusion_matrix(y_test, y_pred) # rows = true malignant, benign
    rows.append({"model": name,
                 "CV accuracy": round(search.best_score_, 3),
                 "test accuracy": round(accuracy_score(y_test, y_pred), 3),
                 "malignant recall": round(recall_score(y_test, y_pred, pos_label=0), 3),
                 "cancers missed": cm[0, 1],     # true malignant, predicted benign
                 "false alarms": cm[1, 0]})      # true benign, predicted malignant
    
results = pd.DataFrame(rows).set_index("model")
print("\n" + results.to_string())
print("Always benign baseline accuracy:", round(np.mean(y_test==1), 3))