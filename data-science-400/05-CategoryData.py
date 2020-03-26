"""
Assignment 5 Categorical Data
Operations:
Normalize numeric values, Bin numeric variables,
Decode categorical data, Impute missing categories, Consolidate categorical data,
One-hot encode categorical data with at least 3 categories,
Remove obsolete columns, Present plots for 1 or 2 categorical columns.
"""
##########################################################

# 1. Import statements for necessary package(s).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

##########################################################

# 2. Read in the dataset from a freely and easily available source on the internet.

# Use Adult Dataset available from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Adult
# This dataset satisfies the following requirements:
# -categorical and numerical attribute types (Mixed)
# -missing data
# -downloadable data file
# -column heading information

# Import data and assign column names as specified in dataset info
# (Will use the training set for this assignment, consisting of 32561 observations)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
Adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                 'hours-per-week', 'native-country','class']

# Additional observations:
# From dataset info, the fnlwgt variable is the sampling weight, representing the number of people 
# with similar demographic characteristics. Since fnlwgt is not directly related to the class variable 
# (>50k or <=50k), and assuming we will not be weighting the observations, this variable will be dropped in step 9.

# By inspection, the information contained in eduation-num is essentially the same as education.
# educaiton-num is an encoded version of education and will be decoded in step 5,
# but will ultimately be dropped in step 9 because of duplication. 
Adult['education'].value_counts()
Adult['education-num'].value_counts()
Adult.groupby('education')['education-num'].mean() # values in education-num correspond to categories in education
Adult.groupby('education')['education-num'].std() # std=0, suggesting they are all the same values

##########################################################

# 3. Normalize numeric values (at least 1 column, but be consistent with other numeric data).

# Per dataset info, age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week are continuous;
# they are also of dtype int64, and have no nan values.
Adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].dtypes
Adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].isna().sum().sum()

# fnlwgt and education-num will be dropped in step 9 (see explanation above in step 2) and will not be normalized here.
# Furthermore, education-num is an encoded version of education, and keeping the raw numeric values
# will make it easier to decode in step 5.

# capital-gain and capital-loss have many zero values (>90% of the observations), and will be 
# binned using arbitrary boundaries (see step 4). To preserve the more intrepretable raw values for binning,
# they will not be normalized here.

# The remaining numeric columns to be normalized are age and hours-per-week.

# z-normalize age and hours-per-week
num_cols = ['age', 'hours-per-week']
Adult[num_cols] = Adult[num_cols].astype(float) # prevent DataConversionWarning by StandardScaler
scaler = StandardScaler().fit(Adult[num_cols])
Adult[num_cols] = scaler.transform(Adult[num_cols])

# check z-normalization results
print('numeric variables after z-normalization:')
for i, (x1, x2) in enumerate(zip(Adult[num_cols].mean(), Adult[num_cols].std())):
    print(Adult[num_cols].columns[i], ': mean=', np.round(x1,1), ',std=', np.round(x2,1))
# mean are all ~0, and std are all ~1, as expected for z-normalization
print('#############################')

##########################################################

# 4. Bin numeric variables (at least 1 column).
    
# Bin capital-gain and capital-loss as they contain many zero values 
Adult[['capital-gain', 'capital-loss']].describe()

# capital-gain
sum(Adult['capital-gain']==0)/len(Adult['capital-gain']) # >91% of observations are zeros
Adult[Adult['capital-gain']>0]['capital-gain'].describe() 
# for values > 0, the mean is 12939; use this as arbitrary bin boundary
MinBin1 = float('-inf')
MaxBin1 = 0
MaxBin2 = Adult[Adult['capital-gain']>0]['capital-gain'].mean() # mean of none-zero values
MaxBin3 = float('inf')
# create binned variable
Adult.loc[(Adult['capital-gain'] > MinBin1) & (Adult['capital-gain'] <= MaxBin1),'capital-gain-bin'] = 'none'
Adult.loc[(Adult['capital-gain'] > MaxBin1) & (Adult['capital-gain'] <= MaxBin2),'capital-gain-bin'] = 'low'
Adult.loc[(Adult['capital-gain'] > MaxBin2) & (Adult['capital-gain'] <= MaxBin3),'capital-gain-bin'] = 'high'
#Adult[['capital-gain','capital-gain-bin']] # check

# capital-loss
sum(Adult['capital-loss']==0)/len(Adult['capital-loss']) # >95% of observations are zeros
Adult[Adult['capital-loss']>0]['capital-loss'].describe() 
# for values > 0, the mean is 1871; use this as arbitrary bin boundary
MinBin1 = float('-inf')
MaxBin1 = 0
MaxBin2 = Adult[Adult['capital-loss']>0]['capital-loss'].mean()  # mean of none-zero values
MaxBin3 = float('inf')
# create binned variable
Adult.loc[(Adult['capital-loss'] > MinBin1) & (Adult['capital-loss'] <= MaxBin1),'capital-loss-bin'] = 'none'
Adult.loc[(Adult['capital-loss'] > MaxBin1) & (Adult['capital-loss'] <= MaxBin2),'capital-loss-bin'] = 'low'
Adult.loc[(Adult['capital-loss'] > MaxBin2) & (Adult['capital-loss'] <= MaxBin3),'capital-loss-bin'] = 'high'
#Adult[['capital-loss','capital-loss-bin']] # check

# while hours-per-week and age could also be binned (e.g. using equal-width bins), they will left 
# as normalized numeric variables.

##########################################################

# 5. Decode categorical data.

# decode education-num, the encoded version of education
coded = Adult['education-num'].value_counts().index # numeric values in education-num
decoded = Adult['education'].value_counts().index # decode to these corresponding categories
for ii in range(0,len(coded)):
    Adult.loc[Adult['education-num'] == coded[ii], 'education-num-decoded'] = decoded[ii]

# check for other variables that require decoding
Adult.dtypes
# Per dataset info, categorical variables are:
# workclass, education, martial-status, occupatuion, relationship, race, sex, native-country, class

# find numeric strings e.g. '3' in these variables that need to be decoded
[x for x in Adult['workclass'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['education'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['martial-status'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['occupation'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['relationship'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['race'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['sex'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['native-country'].unique() if x.lstrip('-+').isnumeric()]
[x for x in Adult['class'].unique() if x.lstrip('-+').isnumeric()]
# By inpsection, there are no other categorical variables with nuemric categories that need to be decoded.

##########################################################
    
# 6. Impute missing categories.

# find categories with missing values with placeholder ' ?'
print('number of missing values/placeholders for categorical variables')
for xx in range(0,Adult.shape[1]):
    if Adult.dtypes[xx].name == 'object':
        print(Adult.columns[xx], ':', sum(Adult[Adult.columns[xx]] == " ?"))
print('#############################')        
# missing values are in workclass, occupation, native-country

# workclass
# counts for each value
Adult['workclass'].value_counts()
# most frequent value
Adult['workclass'].value_counts().index[0]
# impute missing values with most frequent value
Adult.loc[Adult['workclass'] == " ?", 'workclass'] = Adult['workclass'].value_counts().index[0]
#Adult['workclass'].value_counts()

 #occupation
Adult['occupation'].value_counts()
# most frequent value
Adult['occupation'].value_counts().index[0]
# impute missing values with most frequent value
Adult.loc[Adult['occupation'] == " ?", 'occupation'] = Adult['occupation'].value_counts().index[0]
#Adult['occupation'].value_counts()

# native-country
Adult['native-country'].value_counts()
# most frequent value
Adult['native-country'].value_counts().index[0]
# impute missing values with most frequent value
Adult.loc[Adult['native-country'] == " ?", 'native-country'] = Adult['native-country'].value_counts().index[0]
#Adult['native-country'].value_counts()

##########################################################

# 7. Consolidate categorical data (at least 1 column).

# determine number of categories in categorical variables
print('number of categories for categorical variables:')
for xx in range(0,Adult.shape[1]):
    if Adult.dtypes[xx].name == 'object':
        print(Adult.columns[xx], ':', Adult.nunique()[xx])
print('#############################')        

# simplify and consolidate native-country, which has the most categories, by grouping into continents
# create lists of the consolidated groups
# guess: South (= South Korea?) and Hong (= Hong Kong?) assigned to Asia
countries = Adult['native-country'].value_counts().index.tolist()
Americas = [countries[ii] for ii in [0, 1, 4, 5, 6, 8, 10, 14, 16, 19, 21, 24, 25, 28, 31, 36, 38]]
Asia = [countries[ii] for ii in [2, 7, 11, 12, 15, 17, 20, 22, 30, 32, 33, 34]]
Europe = [countries[ii] for ii in [3, 9, 13, 18, 23, 26, 27, 29, 35, 37, 39, 40]]
continent = {'Americas':Americas, 'Asia':Asia, 'Europe':Europe}
for key in continent.keys():
    for countries in continent[key]:
        Adult.loc[Adult.loc[:,'native-country'] == countries , 'continent'] = key
# check the value counts
Adult['continent'].value_counts()

# simplify and consolidate workclass
# create lists of the consolidated groups
classes = Adult['workclass'].value_counts().index.tolist()
Private = [classes[ii] for ii in [0]]
Government = [classes[ii] for ii in [2, 3, 5]]
SelfEmp = [classes[ii] for ii in [1, 4]]
NoPay = [classes[ii] for ii in [6, 7]]
workclass = {'Private':Private, 'Government':Government, 'SelfEmp':SelfEmp, 'NoPay':NoPay}
for key in workclass.keys():
    for classes in workclass[key]:
        Adult.loc[Adult.loc[:,'workclass'] == classes , 'workclass-c'] = key
# check the value counts
Adult['workclass-c'].value_counts()

# simplify occupation by consolidating [Other-service, Tech-support, Protective-serv, 
# Priv-house-serv, Armed-Forces] as Service
jobs = Adult['occupation'].value_counts().index.tolist()
Service = [jobs[ii] for ii in [5, 10, 11, 12, 13]]
Adult['occupation-c'] = Adult['occupation']
for xx in Service:
    Adult.loc[Adult.loc[:, 'occupation'] == xx , 'occupation-c'] = ' Service'
Adult['occupation-c'].value_counts()

# simplify martial-status by consolidating 
# [Married-civ-spouse, Married-AF-spouse, Married-spouse-absent] as Married
# [Divorced, Separated, Windowed] as WasMarried
# create lists of the consolidated groups
status = Adult['martial-status'].value_counts().index.tolist()
Married = [status[ii] for ii in [0, 5, 6]]
NeverMarried = [status[ii] for ii in [1]]
WasMarried = [status[ii] for ii in [2, 3, 4]]
martial = {'Married':Married, 'NeverMarried':NeverMarried, 'WasMarried':WasMarried}
for key in martial.keys():
    for status in martial[key]:
        Adult.loc[Adult.loc[:,'martial-status'] == status , 'martial-status-c'] = key
# check the value counts
Adult['martial-status-c'].value_counts()
    
# simplify education (or education-num-decoded) by consolidating
# [Preschool to 12th grade] as PrimSec, [Assoc-voc, Assoc-acdm] as Assoc, [HS-grad, Some-college] as HighSch,
# [Masters, Prof-school, Doctorate] as GradSch
# create lists of the consolidated groups
levels = Adult['education'].value_counts().index.tolist()
HighSch = [levels[ii] for ii in [0, 1]]
Bachelors = [levels[ii] for ii in [2]]
GradSch = [levels[ii] for ii in [3, 9 , 12]]
PrimSec = [levels[ii] for ii in [5, 7, 8, 10, 11, 13, 14, 15]]
Assoc = [levels[ii] for ii in [4, 6]]
edu = {'HighSch':HighSch, 'Bachelors':Bachelors, 'GradSch':GradSch, 
       'PrimSec':PrimSec, 'Assoc':Assoc}
for key in edu.keys():
    for levels in edu[key]:
        Adult.loc[Adult.loc[:,'education'] == levels , 'education-c'] = key
# check the value counts
Adult['education-c'].value_counts()

##########################################################

# 10. Present plots for 1 or 2 categorical columns.
# create plots before performing one-hot-encoding and removing obsolete columns
# step 8, 9 will follow below

# categorical column: education (consolidated; 16 categories into 5)
Adult['education-c'].value_counts().plot(kind='bar')
plt.title('categorical column: education (consolidated)')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.show()

# categorical column: martial-status (consolidated; 7 categories into 3)
Adult['martial-status-c'].value_counts().plot(kind='bar')
plt.title('categorical column: martial-status (consolidated)')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.show()

##########################################################

# 8. One-hot encode categorical data with at least 3 categories (at least 1 column).

# categorical data with at least 3 categories (sex and class have <3 categories)
encode_col = ['workclass-c', 'education-c', 'martial-status-c', 'occupation-c', 'relationship', 'race',
              'capital-gain-bin', 'capital-loss-bin', 'continent']
dummy_prefix = ['workclass', 'edu', 'martial', 'occu', 'rel', 'race', 'capgain',
                'caploss', 'continent']

# remove space in string for consistency: for example, in race variable, from ' White' to 'White'
cols_to_strip = ['occupation-c', 'relationship', 'race', 'sex', 'class']
for xx in cols_to_strip:
    Adult[xx] = Adult[xx].str.lstrip()

# represent all categories except for one to avoid a linearly dependent data set
Adult = pd.get_dummies(Adult, prefix=dummy_prefix, columns=encode_col, drop_first=True)

##########################################################

# 9. Remove obsolete columns.

# drop columns that have been binned 
binned_cols = ['capital-gain', 'capital-loss']
Adult = Adult.drop(binned_cols, axis=1)

# drop columns that have been decoded
Adult = Adult.drop(['education-num'], axis=1)

# drop columns that have been consolidated
consolidated_cols = ['native-country', 'workclass', 'occupation', 'martial-status', 'education']
Adult = Adult.drop(consolidated_cols, axis=1)

# drop duplicate column education-num-decoded (duplicate of education)
Adult = Adult.drop(['education-num-decoded'], axis=1)

# drop fnlwgt (discussed in step 2)
Adult = Adult.drop(['fnlwgt'], axis=1)

print('Processed data:')
print(Adult.dtypes)
print('#############################')

##########################################################

# 11. Add comments to explain the code blocks.

# Please see comments in code blocks above.

##########################################################

# 12. Add a summary comment block on how the categorical variables have been treated: 
# decoded, imputed, consolidated, and one-hot encoded. Which attributes were plotted and why?

# How the categorical variables have been treated: (Please see code blocks above for more detailed explanation)
# Decoded: 'education-num' contained integrers that represented different levels/categories of education and was decoded.
# Imputed: missing values in 'workclass', 'occupation', 'native-country' with placeholder ' ?' 
# were imputed with the most frequent value.
# Consolidated: 'workclass', 'education', 'martial-status', 'occupation', 'native-country' 
# were simplified by consolidating categories.
# In addition, numeric variables 'capital-gain' and 'capital-loss' were binned and converted into categories.
# One-hot encoded: all categorical variables with at least 3 categories were one-hot encoded.

# Which attributes were plotted and why?
# Per instructions (Present plots for 1 or 2 categorical columns), 2 categorical columns were plotted:
# education and martial-status (both after consolidation). They were plotted as examples to show
# the value counts for each consolidated category.

##########################################################
