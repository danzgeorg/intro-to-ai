from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

wine = datasets.load_wine()
X = wine.data
y = wine.target
print("X shape:", X.shape, "| y shape:", y.shape)


#Hold back 25% of the wines as unseen test data. random_state=42 makes the shuffle repeatable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#criterion='entropy' picks each split by information gain
tree = DecisionTreeClassifier(criterion='entropy')
#learn from the training data
tree.fit(X_train, y_train)
#predict labels for the unseen test data
y_pred = tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Hold-out accuracy: %.2f' % acc)

#rows are the true classes, columns are the predicted classes
#the diagonal shows the correct predictions
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=wine.target_names).plot()
plt.show()

#the data is cut into 5 parts. each part is used once as the validation
# set while the other 4 are used for training
# random_state=42 makes the folds repeatable
kf = KFold(5, shuffle=True, random_state=42)
fold = 1

for train_index, validate_index in kf.split(X,y):
    tree.fit(X[train_index], y[train_index])
    #train on 4 folds
    fold_pred = tree.predict(X[validate_index]) #predict the 5th
    fold_acc = accuracy_score(y[validate_index], fold_pred)
    print(f'Fold {fold}: accuracy {fold_acc:.2f}')
    fold += 1