from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR    # SVR lives in sklearn.svm, not in sklearn itself


# load the data
diabetes = datasets.load_diabetes(as_frame=True)
df = diabetes.frame

print(df[:5])
print(df.shape)    # (442, 11)

# features (X) and target (y)
X = diabetes.data.values
y = diabetes.target.values

# split into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# one learned coefficient per feature, because the features are already scaled,
# their sizes can be compared: bigger (positive or negative) = more influence.
print(model.coef_)

# predict the test set
y_pred = model.predict(X_test)

# build a new data frame with two columns, the actual values of the test data
# and the predictions of the model, and show the first 25 side by side
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

# bar chart of those 25: each pair of bars is one patient (actual vs predicted)
df_head.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# evaluate the linear regression
# Mean: the average target, to judge the RMSE against.
# RMSE: typical size of a prediction error, in the target's units (lower is better).
# R2: share of the variance explained, 0 = no better than predicting the mean,
#     1 = perfect (higher is better).
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))

# pearson correlation between predictions and actual values (1 = perfect)
print('Correlation: ', np.corrcoef(y_test, y_pred)[0, 1])

# predicted vs actual
plt.scatter(y_test, y_pred, s=15)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'k--')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Linear regression')
plt.show()

# repeat with a support vector machine for regression (SVR)
# kernel='rbf' allows a curved (nonlinear) fit
svr = SVR(kernel='rbf', C=100).fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print('SVR RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr)))
print('SVR R2:', metrics.r2_score(y_test, y_pred_svr))


# regression chart
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t = t.sort_values(by=['y'])    # (inplace=True is being removed from pandas)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# first 100 of the 111 test patients
chart_regression(y_pred[:100].flatten(), y_test[:100], sort=True)

# conclusion
# Linear regression: RMSE about 56 against a mean of about 153, R2 about 0.36,
# correlation about 0.61. It explains roughly a third of the variation, so
# progression is only partly predictable from these features.
# SVR (rbf, C=100): RMSE about 62, R2 about 0.21, so it did worse than the
# simpler linear model on this split.