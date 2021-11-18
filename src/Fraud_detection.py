
#######################################################################################################################
#Libraries
######################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
#####################################################################################################################


####################################################################################################################
##Import spreadsheet
#######################################################################################################################
df = pd.read_csv(r'C:\Users\Benny\Documents\DataScienceProject\Analytics\fraud_detection_bank_dataset.csv', sep=',')
df= df.drop(["Unnamed: 0"], axis = 1)
df.head()
df.shape
#df.info()
## Dataset has 113 columns and 20468 observations
######################################################################################################################


####################################################################################################################
#Cleaning Data
#####################################################################################################################

# some columns have unique values Zeros
# Removing unique column values
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col,inplace=True,axis=1)
c_mat1 = df.corr()
corr_table = df[df.columns[1:]].corr()['targets'][:]
df.describe()


####Checking for missing values
df.isnull().sum()
df.isnull().sum().sum()
### Hence No missing numbers in the datasets
#####################################################################################################################



#####################################################################################################################
##EDA
#####################################################################################################################
## Characteristic of dependent varaibel

his = df["targets"].hist
plt.hist(df["targets"])
## Its a descrete varible with values 0 and 1

#####  We consruct a correlation matrix for 20 varaibles that correlates highly with the target variable
plt.figure()
corrmat = df.corr()
k = 30 #number of variables for heatmap
cols = corrmat.nlargest(k, 'targets')['targets'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

######################################################################################################################


####### Standardizing the variables
Y = df["targets"]
X = df.drop(["targets"], axis = 1)

# Since the variables have different scales,I used pre-processing from scikit-learn to standardize the variables
from sklearn import preprocessing
x= preprocessing.StandardScaler().fit_transform(X)
x = pd.DataFrame(x)


#######################################################################################################################
###Define new dataframe with standardized varaibles
col = df.columns[0:97]
x.columns = [col]
new_df = pd.concat([Y, x], axis=1)


######################################################################################################################
####Splitting data into train test set
#####################################################################################################################
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(new_df, test_size=0.3, random_state=1)

x_train = df_train.drop(["targets"], axis = 1)
y_train = df_train["targets"]

x_test = df_test.drop(["targets"], axis = 1)
y_test = df_test["targets"]






















