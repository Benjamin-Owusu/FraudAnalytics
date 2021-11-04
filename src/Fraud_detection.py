
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\Users\Benny\Documents\DataScienceProject\Analytics\fraud_detection_bank_dataset.csv', sep=',')
df= df.drop(["Unnamed: 0"], axis = 1)

#df.isnull().sum().sum()
#df.isnull().values.any()
#df[''].isnull().values.any()

df.head()
df.shape
#df.info()
## Dataset has 113 columns and 20468 observations



#### from the correlation matrix, it is obvious that some columns have unique values.
## Removing unique column values
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col,inplace=True,axis=1)

c_matri1 = df.corr()

corr_table = df[df.columns[1:]].corr()['targets'][:]


plt.figure()
corrmat = df.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'targets')['targets'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


####### Standardizing the variables
Y = df["targets"]
X = df.drop(["targets"], axis = 1)

from sklearn import preprocessing  # Since the variables have different scales,I used pre-processing from scikit-learn to standardize the variables
x= preprocessing.StandardScaler().fit_transform(X)
x = pd.DataFrame(x)

new_df = pd.concat([Y, x], axis=1)

#########################################################################
####Splitting data into train test set
########################################################################
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(new_df, test_size=0.3, random_state=1)

x_train = df_train.drop(["targets"], axis = 1)
y_train = df_train["targets"]

x_test = df_test.drop(["targets"], axis = 1)
y_test = df_test["targets"]


##################################################################
######Logistics Regression Model
#################################################################

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(x_train, y_train)

RSquare_log = log_reg.score(x_train, y_train)
print(RSquare_log)

print(mean_squared_error(y_train, log_reg.predict(x_train))) #### MSE in-sample
print(mean_squared_error(y_test, log_reg.predict(x_test)))  #####  MSE out of sample



#########################################################################################
######## (vii) Random forest
#########################################################################################


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100,random_state=0, bootstrap = True)

rd_forest= regr.fit(x_train, y_train)

RSquare_rd = rd_forest.score(x_train,y_train)
print(RSquare_rd)   ###  R-squared

rd_forest.feature_importances_ # Which variable are  more important
feature = list(x_train.columns)
plt.barh(feature, rd_forest.feature_importances_)  # A plot of the most important variable for the prediction
plt.xlabel("Random Forest Feature Importance")

print(mean_squared_error(y_train, rd_forest.predict(x_train))) #### MSE in-sample
print(mean_squared_error(y_test, rd_forest.predict(x_test)))   #####  MSE out of sample

### The Random forest performs better than Logistics Regression. Higher R squared and lowere MSE in-sample and out of sample




##################################################################
### Next, I create more regressors
#################################################################
from sklearn.preprocessing import PolynomialFeatures   # creating polynomial terms and interraction terms
polynomial_features= PolynomialFeatures(degree=2)  # quadratic and interraction terms
X_poly = pd.DataFrame(polynomial_features.fit_transform(x))





#####################################################################
#### Lasso with Logistics Regression
#####################################################################

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l1',
    solver='saga',  # or 'liblinear'
    C=regularization_strength)

model.fit(x, y)
