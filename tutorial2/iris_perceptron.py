from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
print("X shape:", X.shape, "y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# perceptron on the raw data
ppn = Perceptron(max_iter=40, tol=0.001, eta0=1)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
print("Raw accuracy: %.2f" % accuracy_score(y_test, y_pred))

#perceptron on the scaled data
sc = StandardScaler()
#learn mean and std from the training data only
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn_std = Perceptron(max_iter=40, tol=0.001, eta0=1)
ppn_std.fit(X_train_std, y_train)
y_pred_std = ppn_std.predict(X_test_std)
print("Scaled accuracy: %.2f" % accuracy_score(y_test, y_pred_std))

#rows are the true classes, columns are the predicted classes
#the diagonal shows the correct predictions
cm1 = confusion_matrix(y_test, y_pred)
cm2 = confusion_matrix(y_test, y_pred_std)
ConfusionMatrixDisplay(cm1, display_labels=iris.target_names).plot()
plt.title("Perceptron: raw data")
ConfusionMatrixDisplay(cm2, display_labels=iris.target_names).plot()
plt.title("Perceptron: scaled data")
plt.show()

print("Raw confusion matrix:\n", cm1)
print("Scaled confusion matrix:\n", cm2)