import numpy as np
import pandas as pd
import pingouin as pg
from sklearn import linear_model

pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 1000)

# Pulire df di Prolific

df1 = pd.read_csv('Dati_prolific.csv')

## CANCELLARE ##

# Colonne Prolific
df1 = df1.drop(columns=['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'RecordedDate',
                      'Duration (in seconds)', 'Finished', 'RecipientLastName',
                      'RecipientFirstName', 'RecipientEmail', 'ExternalReference',
                      'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
                      'UserLanguage', 'ProlificID', 'PROLIFIC_PID'
                      ])
df1 = df1.drop([0, 1], axis=0)

## DATI GRATIS
df2 = pd.read_csv('Dati_gratis.csv')
# Colonne gratis
df2 = df2.drop(columns=['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'RecordedDate',
                      'Duration (in seconds)', 'Finished', 'RecipientLastName',
                      'RecipientFirstName', 'RecipientEmail', 'ExternalReference',
                      'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
                      'UserLanguage'
                      ])
df2 = df2.drop([0, 1], axis=0)
# Dataframe finale
df = pd.concat([df1, df2])

# Togliamo i click
for col in df.columns:
    if "Click" in col:
        del df[col]

df = df.drop(columns = ['Social_9_TEXT', 'Altraroba', 'Commenti'])

df = df.dropna(axis = 0)
df.head()
df.shape[0]

df.to_csv('Dati.csv')

## RIORGANIZZARE IL DATAFRAME

df = pd.read_csv('Dati.csv')

# Cancellare tutto dopo il trattino " _ "
df.columns = df.columns.str.split('_').str[0]

# Questionari #
# Trasformare in numerico
cols = ['PU1', 'PU2', 'PU3', 'OverU1', 'OverU2', 'OverU3', 'AGRR', 'AGR', 'COS', 'COSR', 'STAB', 'STABR',
        'EX', 'EXR', 'OP', 'OPR']
df[cols] = df[cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)

# Items al contrario
df['AGR2'] = 100 - df['AGRR']
df['COS2'] = 100 - df['COSR']
df['STAB2'] = 100 - df['STABR']
df['EX2'] = 100 - df['EXR']
df['OP2'] = 100 - df['OPR']

# Calcolo dei punteggi
# BIG 5
df['Agreeableness'] = (df['AGR'] + df['AGR2'])/2
df['Conscientiousness'] = (df['COS'] + df['COS2'])/2
df['Em Stability'] = (df['STAB'] + df['STAB2'])/2
df['Extroversion'] = (df['EX'] + df['EX2'])/2
df['Openness'] = (df['OP'] + df['OP2'])/2

# Social media use
df['Social Use'] = (df['PU1'] + df['PU2'] + df['PU3'] + df['OverU1'] + df['OverU2'] + df['OverU3'])/6
df = df.drop(columns=['PU1', 'PU2', 'PU3', 'OverU1', 'OverU2', 'OverU3', 'AGRR', 'AGR', 'COS', 'COSR', 'STAB', 'STABR',
                      'EX', 'EXR', 'OP', 'OPR', 'AGR2', 'COS2', 'STAB2', 'STAB2', 'EX2', 'OP2'])

df = df.drop(columns=['Unnamed: 0'])

## CONTROLLARE ALPHA DI CRONBACH PER SOCIAL MEDIA USE
#cronb = df[['PU1','PU2','PU3','OverU1','OverU2','OverU3']]
#pg.cronbach_alpha(data = cronb, ci = .99)
# Ricordati di riportarla da qualche parte

# Formattare diversamente il dataframe
longdf = pd.melt(frame=df, id_vars=['ResponseId', 'Genere', 'Età', 'Campo', 'Studio', 'Social', 'dx/sx',
                                    'Cons/prog', 'Ind/col', 'Tempo', 'Flame', 'Agreeableness',
                                    'Conscientiousness', 'Em Stability', 'Extroversion',
                                    'Openness', 'Social Use'], var_name="Domanda", value_name="Risposta")



longdf.to_csv('Lungo_df_pulito.csv')

longdf['Topic'] = longdf['Domanda'].apply(lambda x: x.split('.')[0])
longdf['Codicino'] = longdf['Domanda'].apply(lambda x: x.split('.')[1])

# Delete the column that had both
longdf = longdf.drop(columns='Domanda')

result = longdf.pivot(index=['ResponseId', 'Genere', 'Età', 'Campo', 'Studio', 'Social', 'dx/sx',
                                    'Cons/prog', 'Ind/col', 'Tempo', 'Flame',
                                    'Agreeableness', 'Conscientiousness', 'Em Stability', 'Extroversion',
                                    'Openness', 'Social Use', 'Topic'], columns='Codicino', values='Risposta')

result = result.reset_index()
result.to_csv('Dati_puliti.csv')

df = pd.read_csv('Dati_puliti.csv')
#df2 = df[["ALDIS", "ALEM", "IODIS", "IOEM"]]
#df2.corr()
#pearsonr(df2.IOEM, df2.ALEM)


### Ora estraiamo i predittori che sicuro ci serviranno
predittori = df.loc[df['Topic'].str.contains('EUT')]
predittori = predittori[['ResponseId', 'dx/sx', 'Cons/prog', 'Ind/col', 'Flame', 'Agreeableness',
                     'Conscientiousness', 'Em Stability', 'Extroversion', 'Openness',
                     'Social Use']]


predittori = predittori.sort_values(by=['ResponseId'], ascending=False)
predittori.head()

predittori.to_csv('Predittori.csv')

### PROIEZIONE SOCIALE: IL CODICE DI ROB

# togliere colonne che non servono
df.head()
df2 = df[["ResponseId", "Topic", "ALDIS", "ALEM", "IODIS", "IOEM"]]

# Separate dataframes for each subject
lista_dicts = []
for name in set(df2['ResponseId']):
    temp_df = df2.loc[df2['ResponseId'] == name]
    temp_df = temp_df.drop(columns=['ResponseId'])
    dict_df = {'Name': name, 'Df': temp_df}
    lista_dicts.append(dict_df)

core = pd.DataFrame(columns=['Name', 'Corr'])

# Append columns for each dataframe
for subject in lista_dicts:
    sub_df = subject.get("Df")
    s1a = sub_df['ALDIS']
    s1b = sub_df['ALEM']
    s2a = sub_df['IODIS']
    s2b = sub_df['IOEM']
    df0 = pd.DataFrame({'A': pd.concat([s1a, s1b]), 'I': pd.concat([s2a, s2b])})
    subject["Df"] = df0
    subject["Corr"] = np.corrcoef(df0['A'], df0['I'])[0, 1]
    core.loc[len(core.index)] = [subject.get("Name"), subject.get("Corr")]

core = core.sort_values(by=['Name'], ascending=False)
core.to_csv('core.csv')

#check delle dimensioni
core.shape[0]
predittori.shape[0]

## U N I R E
core = pd.read_csv('core.csv')
predittori = pd.read_csv('Predittori.csv')

lasso_df = pd.concat([core, predittori], axis=1)
lasso_df.head()

lasso_df = lasso_df.drop(columns = ['ResponseId', 'Unnamed: 0'])

lasso_df.to_csv('DatixAnalisi.csv')

lasso_df = pd.read_csv('DatixAnalisi.csv')
lasso_df.head()










### IT'S LASSING TIME ###
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

#define predictor and response variables
X = lasso_df[["dx/sx", "Cons/prog", "Ind/col", "Flame", "Agreeableness", "Conscientiousness", "Em Stability", "Extroversion", "Openness", "Social Use"]]
y = lasso_df[["Corr"]]

from sklearn.pipeline import make_pipeline

#scaler = StandardScaler()
#X_sc = scaler.fit_transform(X)
#Y_sc = scaler.fit_transform(y)

# Split the data into 33% test and 77% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
# fitting the training data
LR.fit(X_train,y_train)
y_prediction =  LR.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)

print('r2 score is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))


## LASSO
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])

search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)

search.best_params_

coefficients = search.best_estimator_.named_steps['model'].coef_

importance = np.abs(coefficients)










#Fitting the model
model = Lasso(alpha= 0.01, fit_intercept=False, positive= True, selection='random')
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error,r2_score
print(r2_score(Y_test,y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(Y_test-y_pred)

param = {
    'alpha':[.00001, 0.0001,0.001, 0.01],
    'fit_intercept':[True,False],
    'positive':[True,False],
    'selection':['cyclic','random'],
    }

#define model
model = Lasso()

# define search
#search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1)

# execute search
result = search.fit(X_sc, Y_sc)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

model = Lasso(alpha=0.01,fit_intercept= True, normalize = True, positive= True, selection = 'cyclic')

model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(r2_score(Y_test,y_pred))




# iniziamo con una regressione multipla
lasso_df.corr()
# Crea l'oggetto regressione lineare
reg = linear_model.LinearRegression()

# per trainare il modello serve fit
reg.fit(lasso_df[['dx/sx', 'Cons/prog', 'Ind/col', 'Flame', 'Agreeableness', 'Conscientiousness',
            'Em Stability', 'Extroversion', 'Openness', 'Social Use']], lasso_df.Corr)

reg.coef_
