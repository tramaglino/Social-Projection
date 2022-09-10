import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
## preparing data
#data_full = pd.read_csv('DatixAnalisi.csv')
data_full = pd.read_csv('Dati_bellissimi.csv')

#define predictor and response variables
X = data_full[["dx/sx", "Cons/prog", "Ind/col", "Flame", "Agreeableness", "Conscientiousness", "Em Stability", "Extroversion", "Openness", "Social Use"]]
y = data_full[["Corr"]]

#Scaling everything
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
Y_sc = scaler.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X_sc, Y_sc, test_size=0.1)

param = {
    'alpha': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
    'fit_intercept':[True,False],
    'positive':[True,False],
    'selection':['cyclic','random'],
    }

#define model
model = Lasso()

# define search
search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1)

# execute search
result = search.fit(X_train, Y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Best model
lasso_best = Lasso(alpha=0.1, fit_intercept=False, positive=True, selection='random').fit(X_train,Y_train)
y_pred = lasso_best.predict(X_test)
print(r2_score(Y_test,y_pred))

root_mean_squared_error = np.sqrt(mean_squared_error(Y_test,y_pred))
final_coeffs = pd.Series(lasso_best.coef_, index = X.columns)
