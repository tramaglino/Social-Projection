from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
#
# Load the Boston Data Set
#
bh = datasets.load_boston()
X = bh.data
y = bh.target
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Create an instance of Lasso Regression implementation
#
lasso = Lasso(alpha=1.0)
#
# Fit the Lasso model
#
lasso.fit(X_train, y_train)
#
# Create the model score
#
lasso.score(X_test, y_test), lasso.score(X_train, y_train)

lasso.coef_