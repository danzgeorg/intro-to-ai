import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations


# raw a confusion matrix as a coloured grid.
# rows are the true species, columns are the predicted species.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load the data into a DataFrame
iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=np.append(iris['feature_names'], ['target']))

# write out a reminder of the shape of the data
print(df.head())

# split into features (X) and target (y)
result = []
for x in df.columns:
    if x != 'target':
        result.append(x)

X = df[result].values
y = df['target'].values
species = iris.target_names    # ['setosa', 'versicolor', 'virginica']

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# visualise the data: every pair of features against each other
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
    ax.scatter(X[:, i], X[:, j], c=y, cmap='viridis', s=15)
    ax.set_xlabel(iris.feature_names[i]); ax.set_ylabel(iris.feature_names[j])
plt.tight_layout(); plt.show()

# build and train the SVM
svm_model = SVC(kernel='linear', C=100, decision_function_shape='ovo').fit(X_train, y_train)

# pedict the species of the test flowers and evaluate
y_pred = svm_model.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# side by side sample of actual vs predicted species (numbers turned back into names)
print(pd.DataFrame({'Actual': iris.target_names[y_test.astype(int)],
                    'Predicted': iris.target_names[y_pred.astype(int)]}).head(15))

# confusion matrix: counts of true species (rows) vs predicted species (columns).
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

# normalised confusion matrix: divide each row by its total, so each row shows the
# fraction of that species predicted as each class (1.0 on the diagonal = all correct).
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, species, title='Normalized confusion matrix')
plt.show()

# all the settings (hyperparameters) the model used, including the defaults
print(svm_model.get_params())
