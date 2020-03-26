"""
Milestone 3: Final Data Model
1. Split your dataset into training and testing sets
2. Train your classifiers, using the training set partition
3. Apply your (trained) classifiers on the test set
4. Measure each classifier's performance using at least 3 of the metrics we covered in this course (one of them has to be the ROC-based one). At one point, you'll need to create a confusion matrix.
5. Document your results and your conclusions, along with any relevant comments about your work
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import *

##########################################################################
# Import cleaned up data, using code developed for Milestone 2
##########################################################################

def prepareAdultData():
    print('Begin data import and preparation...')
    print('#############################')          
    # Read in the dataset from a freely and easily available source on the internet.
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
    Adult_original = Adult.copy()
    # Additional observations:
    # From dataset info, the fnlwgt variable is the sampling weight, representing the number of people 
    # with similar demographic characteristics. Since fnlwgt is not directly related to the class variable 
    # (>50k or <=50k), and assuming we will not be weighting the observations, this variable will be dropped.
    
    # By inspection, the information contained in eduation-num is essentially the same as education.
    # educaiton-num is an encoded version of education and will be decoded,
    # but will ultimately be dropped because of duplication. 
    Adult['education'].value_counts()
    Adult['education-num'].value_counts()
    Adult.groupby('education')['education-num'].mean() # values in education-num correspond to categories in education
    Adult.groupby('education')['education-num'].std() # std=0, suggesting they are all the same values
    
    ##########################################################
    
    # Normalize numeric values.
    
    # Per dataset info, age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week are continuous;
    # they are also of dtype int64, and have no nan values.
    Adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].dtypes
    Adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].isna().sum().sum()
    
    # fnlwgt and education-num will be dropped and will not be normalized here.
    # Furthermore, education-num is an encoded version of education, and keeping the raw numeric values
    # will make it easier to decode.
    
    # capital-gain and capital-loss have many zero values (>90% of the observations), and will be 
    # binned using arbitrary boundaries. To preserve the more intrepretable raw values for binning,
    # they will not be normalized here.
    
    # The remaining numeric columns to be normalized are age and hours-per-week.
    
    # New: Replace outliers before normalizing
    # referencing the histograms created above, and without additional domain knowledge,
    # use a standard definition for outliers: x > mean(x)+2*std(x) or x < mean(x)-2*std(x)
    # define function that replaces outliers with the median 
    def replace_outlier(y):
        x = np.copy(y)
        LimitHi = np.mean(x) + 2*np.std(x)
        LimitLo = np.mean(x) - 2*np.std(x)
        # Create Flags for outliers 
        FlagBad = (x < LimitLo) | (x > LimitHi)
        # Replace outliers with the median
        x = x.astype(float) # since np.median returns float
        x[FlagBad] = np.median(x)
        return x
    
    # perform replacement on numeric columns
    num_cols = ['age', 'hours-per-week']
    Adult[num_cols] = Adult[num_cols].apply(replace_outlier)
    
    #check replacement:
    sum(Adult['age']==np.median(Adult['age']))
    sum(Adult_original['age']==np.median(Adult_original['age']))+sum(Adult_original['age']>np.mean(Adult_original['age'])+2*np.std(Adult_original['age']))+sum(Adult_original['age']<np.mean(Adult_original['age'])-2*np.std(Adult_original['age']))
    
    sum(Adult['hours-per-week']==np.median(Adult['hours-per-week']))
    sum(Adult_original['hours-per-week']==np.median(Adult_original['hours-per-week']))+sum(Adult_original['hours-per-week']>np.mean(Adult_original['hours-per-week'])+2*np.std(Adult_original['hours-per-week']))+sum(Adult_original['hours-per-week']<np.mean(Adult_original['hours-per-week'])-2*np.std(Adult_original['hours-per-week']))
    
    # Min-Max normalize age and hours-per-week
    num_cols = ['age', 'hours-per-week']
    #Adult[num_cols] = Adult[num_cols].astype(float) # prevent DataConversionWarning by scaler
    scaler = MinMaxScaler().fit(Adult[num_cols])
    Adult[num_cols] = scaler.transform(Adult[num_cols])
    
#    # check normalization results
#    print('numeric variables after Min Max normalization:')
#    for i, (x1, x2) in enumerate(zip(Adult[num_cols].max(), Adult[num_cols].min())):
#        print(Adult[num_cols].columns[i], ': max=', np.round(x1,1), ',min=', np.round(x2,1))
#    print('#############################')
    
    ##########################################################
    
    # Bin numeric variables.
        
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
    
    # New: make one capital categorical var
    #Adult[(Adult['capital-gain']!=0) & (Adult['capital-loss']!=0)]
    Adult.loc[Adult['capital-loss-bin']=='low', 'capital-bin'] = 'LowLoss'
    Adult.loc[Adult['capital-loss-bin']=='high', 'capital-bin'] = 'HighLoss'
    Adult.loc[Adult['capital-gain-bin']=='low', 'capital-bin'] = 'LowGain'
    Adult.loc[Adult['capital-gain-bin']=='high', 'capital-bin'] = 'HighGain'
    Adult.loc[(Adult['capital-gain-bin']=='none') & (Adult['capital-loss-bin']=='none'), 'capital-bin'] = 'NoChange'
    
    #check:
    Adult['capital-bin'].value_counts()
    Adult['capital-loss-bin'].value_counts()
    Adult['capital-gain-bin'].value_counts()
        
    ##########################################################
    
    # Decode categorical data.
    
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
        
    # Impute missing categories.
    
#    # find categories with missing values with placeholder ' ?'
#    print('number of missing values/placeholders for categorical variables')
#    for xx in range(0,Adult.shape[1]):
#        if Adult.dtypes[xx].name == 'object':
#            print(Adult.columns[xx], ':', sum(Adult[Adult.columns[xx]] == " ?"))
#    print('#############################')        
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
    
    # Consolidate categorical data.
    
#    # determine number of categories in categorical variables
#    print('number of categories for categorical variables:')
#    for xx in range(0,Adult.shape[1]):
#        if Adult.dtypes[xx].name == 'object':
#            print(Adult.columns[xx], ':', Adult.nunique()[xx])
#    print('#############################')        
    
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
    
    # simplify and consolidate occupation 
    # create lists of the consolidated groups
    jobs = Adult['occupation'].value_counts().index.tolist()
    Specialty = [jobs[ii] for ii in [0]]
    BlueCollar = [jobs[ii] for ii in [1, 6, 7, 8, 9]]
    WhiteCollar = [jobs[ii] for ii in [2, 3]]
    Sales = [jobs[ii] for ii in [4]]
    Service = [jobs[ii] for ii in [5, 10, 11, 12, 13]]
    occupation = {'Specialty':Specialty, 'BlueCollar':BlueCollar, 'WhiteCollar':WhiteCollar, 'Sales':Sales, 'Service':Service}
    for key in occupation.keys():
        for jobs in occupation[key]:
            Adult.loc[Adult.loc[:,'occupation'] == jobs , 'occupation-c'] = key
    # check the value counts
    Adult['occupation-c'].value_counts()
    
    # simplify and consolidate martial-status
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
        
    # simplify and consolidate education
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
        
    # inspect original and processed data:
    Adult['workclass-c'].value_counts()  
    Adult['education-c'].value_counts() 
    Adult['martial-status-c'].value_counts() 
    Adult['occupation-c'].value_counts() 
    Adult['relationship'].value_counts() 
    Adult['race'].value_counts()
    Adult['sex'].value_counts() 
    Adult['capital-bin'].value_counts() 
    Adult['continent'].value_counts() 
    Adult['class'].value_counts() 
    
    Adult_original['workclass'].value_counts()  
    Adult_original['education'].value_counts() 
    Adult_original['martial-status'].value_counts() 
    Adult_original['occupation'].value_counts() 
    Adult_original['relationship'].value_counts() 
    Adult_original['race'].value_counts()
    Adult_original['sex'].value_counts() 
    Adult_original['class'].value_counts() 
    
    ##########################################################    
    
    # One-hot encode categorical data 
    
    encode_col = ['workclass-c', 'education-c', 'martial-status-c', 'occupation-c', 'relationship', 'race',
                  'sex', 'capital-bin', 'continent', 'class']
    dummy_prefix = ['workclass', 'edu', 'martial', 'occu', 'rel', 'race', 'sex',
                    'capital', 'continent', 'class']
    
    # remove space in string for consistency: for example, in race variable, from ' White' to 'White'
    cols_to_strip = ['relationship', 'race', 'sex', 'class']
    for xx in cols_to_strip:
        Adult[xx] = Adult[xx].str.lstrip()
    
    # represent all categories except for one to avoid a linearly dependent data set
    Adult = pd.get_dummies(Adult, prefix=dummy_prefix, columns=encode_col, drop_first=True)
    
    ##########################################################
    
    # Remove obsolete columns.
    
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
        
    # capital-gain and loss bins have been combined
    cap_cols = ['capital-gain-bin', 'capital-loss-bin']
    Adult = Adult.drop(cap_cols, axis=1)
    
    ##########################################################
    
    print('Processed data:')
    print(Adult.dtypes)
    print('#############################')
    print('End data import and preparation...')

    # return dictionary of original and processed dataset
    return {'Adult':Adult,'Adult_original':Adult_original}

##########################################################
# load processed dataset
Adult_df = prepareAdultData()['Adult']

##########################################################################
# Short narrative on the data preparation for chosen data set from Milestone 2.
##########################################################################

# -number of observations and attributes
# observations: 32561
# attributes: 12 original attributes, 
# expanded to 32 after one-hot encoding of categorical variables (with all categories except for one)
#Adult_df.info()
# -datatype, distribution, and a comment on each attribute
# attributes:
# 1. age: numeric, see histogram for distribution, represents age 
# outliers (2*std) removed and min-max normalized
#Adult_df['age'].describe()
#Adult_df['age'].hist() # most values around 0.5 after min-max normalization
#plt.ylabel('count')
#plt.title('histogram of age')
#plt.show()
# 2. hours-per-week: numeric, see histogram for distribution, represents hours worked per week
# outliers (2*std) removed and min-max normalized
#Adult_df['hours-per-week'].describe()
#Adult_df['hours-per-week'].hist() # most values around 0.5 after min-max normalization
#plt.ylabel('count')
#plt.title('histogram of hours-per-week')
#plt.show()
# 3. workclass: one-hot encoded categorical attribute, binary distribution, represents workclass
# levels: Government, Private, SelfEmployed, NoPay
# 4. edu: one-hot encoded categorical attribute, binary distribution, represents education level
# levels: Primary/Secondary, HighSchool, Bachelors, Associate, GradSchool
# 5. martial: one-hot encoded categorical attribute, binary distribution, represents martial status
# levels: Married, WasMarried, NeverMarried
# 6. occu: one-hot encoded categorical attribute, binary distribution, represents occupation
# levels: Sales, Service, Specialty, WhiteColar, BlueColar
# 7. rel: one-hot encoded categorical attribute, binary distribution, represents relationship
# levels: Not in family, Other relative, Own child, Husband, Wife
# 8. race: one-hot encoded categorical attribute, binary distribution, represents race
# levels: Asian/Pacific Islander, Black, White, Other, American Indian
# 9. sex: one-hot encoded categorical attribute, binary distribution, represents sex
# levels: Male, Female
# 10. capital: one-hot encoded categorical attribute, binary distribution, represents capital (new category concatenated from others)
# levels: NoChange, LowLoss, HighLoss, LowGain, HighGain
# 11. continent: one-hot encoded categorical attribute, binary distribution, represents continent (new category from native country)
# levels: Americas, Europe, Asia
# 12. class_>50K: one-hot encoded categorical attribute, binary distribution, represents income >50K
# levels: >50K, <=50K

# -Source citation for your data set
# Adult Dataset available from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Adult

# please see comments embedded in above code blocks for detailed data prepration steps

##########################################################################
# Ask a binary-choice question that describes your classification. 
# Write the question as a comment. 
# Specify an appropriate column as your expert label for a classification (include decision comments).
##########################################################################

# Binary-choice question: Does income exceed $50k per year? 
# Expert label column: class_>50K
# Decision comment: class_>50K is binary, with 1 and 0 representing income >50K and <=50K a year, respectively,
# and is appropriate for this question.

##########################################################################
# Apply K-Means on some of your columns, but make sure you do not use the expert label. 
# Add the K-Means cluster labels to your dataset.
##########################################################################

print('\nApply K-Means on some columns, and add the K-Means cluster labels to dataset...')

# X: new copy of data with the selected attributes for clustering
X = Adult_df.copy()[['age', 'hours-per-week', 'workclass_NoPay', 'workclass_Private', 'workclass_SelfEmp',
                 'sex_Male']]

# estimate optimal number of clusters; compare within-cluster sum of squares for k=2 to k=7
wcss = []
for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#plt.plot(range(2,8), wcss)
#plt.xlabel('Number of clusters')
#plt.ylabel('Within-cluster sum of squares')
#plt.title('find optimal number of clusters')
#plt.show()

# pick k = 3 based on the elbow method (4 could also work but will try 3 here)
kmeans = KMeans(n_clusters=3)
Y = kmeans.fit_predict(X)

# Add the cluster label to the dataset.
Adult_df['cluster'] = Y
# to be consistent with other columns, one-hot encode cluster label, 
# represent all categories except for one to avoid a linearly dependent data set
Adult_df = pd.get_dummies(Adult_df, prefix='cluster', columns=['cluster'], drop_first=True)

##########################################################################
# Milestone 3 Task 1
# Split your dataset into training and testing sets
##########################################################################

X = Adult_df.copy().drop('class_>50K', axis=1) #features
y = Adult_df.copy()['class_>50K'] #expert label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Decision comment: use most of the data for training. A 80/20 train/test split is often reasonable.

##########################################################################
# Classifier 1: logistic regression 
##########################################################################
print ('\n\nClassifier 1: Logistic regression classifier')
print ('################################################')

##########################################################################
# Milestone 3 Task 2
# Train your classifiers, using the training set partition
##########################################################################

classifier = LogisticRegression(solver='saga') 
classifier.fit(X_train, y_train) 
# Decision comment: a logistic regression classifier is well-suited for binary classification problems,
# returning probability estimates that could be used to predict class labels.
# specified saga solver, otherwise using default parameters

##########################################################################
# Milestone 3 Task 3
# Apply your (trained) classifiers on the test set
##########################################################################

y_test_prob = classifier.predict_proba(X_test) # probability estimates
y_test_pred = classifier.predict(X_test) # predict class labels

##########################################################################
# Milestone 3 Task 4
# Measure each classifier's performance using at least 3 of the metrics we covered in this course 
# (one of them has to be the ROC-based one). At one point, you'll need to create a confusion matrix.
##########################################################################

# as reference, print confusion matrix based on default 0.5 probability threshold
print ('\nAs reference, Confusion Matrix using default 0.5 probability threshold:')
print(confusion_matrix(y_test, y_test_pred)) 

# Specify and justfify the probability threshold:
# Choosing the threshold is problem-dependent. For example, one could consider 
# the consequences (cost) of false positives or false negatives. 
# Here, for predicting whether income >50K per year, the ROC curve was used to help determine 
# an appropriate probability threshold. Specifically, a threshold that is based on the point in an ROC curve
# that is closest to the top-left corner was used.

# test a few thresholds from 0 to 1
test_thrs = np.arange(0, 1.05, 0.05)
temp_tprs = np.array([])
temp_fprs = np.array([])
for test_thr in test_thrs:
    # predict class labels based on custom threshold, using prob of classification for outcome/class 1
    temp_prediction = (y_test_prob[:,1]>test_thr).astype(int) 
    cm_temp = confusion_matrix(y_test, temp_prediction) # get confusion matrix
    tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel() 
    temp_tprs = np.append(temp_tprs, tp_temp/(tp_temp+fn_temp)) # get tprs to plot ROC curve
    temp_fprs = np.append(temp_fprs, fp_temp/(fp_temp+tn_temp)) # get fprs to plot ROC curve

# plot test/partial ROC curve
plt.figure(figsize=[6,6])
plt.plot(temp_fprs, temp_tprs,'o-')
for i, x in enumerate(zip(temp_fprs, temp_tprs)):
    plt.text(x[0]+0.03, x[1]-0.02, test_thrs[i])
plt.title('test/partial ROC curve \nto help determine probability threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.text(np.min(temp_fprs), np.max(temp_tprs), 'labels: thresholds')
plt.show()

# opt for the threshold that is based on the point in an ROC curve that is closest to the top-left corner
dist_top_left = np.sqrt((temp_tprs-1)**2+(temp_fprs-0)**2) # distance to top-left corner
thr = np.round(test_thrs[np.argmin(dist_top_left)],2) # specify custom threshold closest to top-left corner
thr1 = thr #store as thr1

# predict class labels based on custom threshold, using prob of classification for outcome/class 1
y_test_pred_custom = (y_test_prob[:,1]>thr).astype(int)
# confusion matrix from predicted values
cm_custom = confusion_matrix(y_test, y_test_pred_custom)
print ('\nConfusion Matrix using probability threshold {}:'.format(thr))
print(cm_custom)
tn, fp, fn, tp = cm_custom.ravel()
print ('\nTP, TN, FP, FN:', tp, ',', tn, ',', fp, ',', fn)

##########################################################################
# Precision, Recall, and F1 measures based on Confusion Matrix.
##########################################################################

print ('\nMetrics based on confusion matrix (for class 1):')
precision = precision_score(y_test, y_test_pred_custom)
print ("Precision:", np.round(precision, 2))
recall = recall_score(y_test, y_test_pred_custom)
print ("Recall:", np.round(recall, 2))
f1 = f1_score(y_test, y_test_pred_custom)
print ("F1:", np.round(f1, 2))
print ('\nFor reference:')
accuracy = accuracy_score(y_test, y_test_pred_custom)
print ("Accuracy:", np.round(accuracy, 2))

print('\ncheck against classification report:')
print(classification_report(y_test, y_test_pred_custom))

##########################################################################
# Calculate the ROC curve and it's AUC 
##########################################################################

# get false positive rate, true posisive rate, probability thresholds
fpr, tpr, thrs = roc_curve(y_test, y_test_prob[:,1])
# get area under the curve
area_under_curve = auc(fpr, tpr)

# plot ROC curve with AUC score
plt.figure(figsize=[6,6])
plt.title('ROC curve\n'+type(classifier).__name__)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1], c='gray', linestyle='--', label='random classifier')
plt.legend(loc="lower right")
plt.show()

print ("\nAUC score (using auc function):", np.round(area_under_curve, 2))
print ("AUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_test_prob[:,1]), 2), "\n")

##########################################################################
# Classifier 2: naive Bayes 
##########################################################################
print ('\n\nClassifier 2: Naive Bayes classifier')
print ('################################################')

##########################################################################
# Milestone 3 Task 2
# Train your classifiers, using the training set partition
##########################################################################

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
# Decision comment: a gaussian naive bayes classifier is suited for binary classification problems
# using default parameters

##########################################################################
# Milestone 3 Task 3
# Apply your (trained) classifiers on the test set
##########################################################################

y_test_prob2 = clf2.predict_proba(X_test) # probability estimates
y_test_pred2 = clf2.predict(X_test) # predict class labels

##########################################################################
# Milestone 3 Task 4
# Measure each classifier's performance using at least 3 of the metrics we covered in this course 
# (one of them has to be the ROC-based one). At one point, you'll need to create a confusion matrix.
##########################################################################

# as reference, print confusion matrix based on default 0.5 probability threshold
print ('\nAs reference, Confusion Matrix using default 0.5 probability threshold:')
print(confusion_matrix(y_test, y_test_pred2)) 
print(classification_report(y_test, y_test_pred2))

# Specify and justfify the probability threshold:
# The ROC curve was used to help determine an appropriate probability threshold. 
# Specifically, a threshold that is based on the point in an ROC curve that is closest to the top-left corner was used.

# test a few thresholds from 0 to 1
test_thrs = np.arange(0, 1.05, 0.05)
temp_tprs = np.array([])
temp_fprs = np.array([])
for test_thr in test_thrs:
    # predict class labels based on custom threshold, using prob of classification for outcome/class 1
    temp_prediction = (y_test_prob2[:,1]>test_thr).astype(int) 
    cm_temp = confusion_matrix(y_test, temp_prediction) # get confusion matrix
    tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel() 
    temp_tprs = np.append(temp_tprs, tp_temp/(tp_temp+fn_temp)) # get tprs to plot ROC curve
    temp_fprs = np.append(temp_fprs, fp_temp/(fp_temp+tn_temp)) # get fprs to plot ROC curve

# plot test/partial ROC curve
plt.figure(figsize=[6,6])
plt.plot(temp_fprs, temp_tprs,'o-')
for i, x in enumerate(zip(temp_fprs, temp_tprs)):
    plt.text(x[0]+0.03, x[1]-0.02, test_thrs[i])
plt.title('test/partial ROC curve \nto help determine probability threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.text(np.min(temp_fprs), np.max(temp_tprs), 'labels: thresholds')
plt.show()

# opt for the threshold that is based on the point in an ROC curve that is closest to the top-left corner
dist_top_left = np.sqrt((temp_tprs-1)**2+(temp_fprs-0)**2) # distance to top-left corner
thr = np.round(test_thrs[np.argmin(dist_top_left)],2) # specify custom threshold closest to top-left corner
thr2 = thr # store as thr2

# predict class labels based on custom threshold, using prob of classification for outcome/class 1
y_test_pred2_custom = (y_test_prob2[:,1]>thr).astype(int)
# confusion matrix from predicted values
cm_custom2 = confusion_matrix(y_test, y_test_pred2_custom)
print ('\nConfusion Matrix using probability threshold {}:'.format(thr))
print(cm_custom2)
tn, fp, fn, tp = cm_custom2.ravel()
print ('\nTP, TN, FP, FN:', tp, ',', tn, ',', fp, ',', fn)

# The confusion matrix sheds light on different aspects of the classifier's performance at the selected probability threshold.
# As such, the confusion matrix is only valuable when its probability threshold can be justified.

##########################################################################
# Precision, Recall, and F1 measures based on Confusion Matrix.
##########################################################################

print ('\nMetrics based on confusion matrix (for class 1):')
precision2 = precision_score(y_test, y_test_pred2_custom)
print ("Precision:", np.round(precision2, 2))
recall2 = recall_score(y_test, y_test_pred2_custom)
print ("Recall:", np.round(recall2, 2))
f12 = f1_score(y_test, y_test_pred2_custom)
print ("F1:", np.round(f12, 2))
print ('\nFor reference:')
accuracy2 = accuracy_score(y_test, y_test_pred2_custom)
print ("Accuracy:", np.round(accuracy2, 2))

print('\ncheck against classification report:')
print(classification_report(y_test, y_test_pred2_custom))

##########################################################################
# Calculate the ROC curve and it's AUC 
##########################################################################

# get false positive rate, true posisive rate, probability thresholds
fpr, tpr, thrs = roc_curve(y_test, y_test_prob2[:,1])
# get area under the curve
area_under_curve = auc(fpr, tpr)

# plot ROC curve with AUC score
plt.figure(figsize=[6,6])
plt.title('ROC curve\n'+type(clf2).__name__)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1], c='gray', linestyle='--', label='random classifier')
plt.legend(loc="lower right")
plt.show()

print ("\nAUC score (using auc function):", np.round(area_under_curve, 2))
print ("AUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_test_prob2[:,1]), 2), "\n")

##########################################################################
# Mileston3 Task 5
# Document your results and your conclusions, along with any relevant comments about your work
##########################################################################

print ('\n\nOverall results and conclusions')
print ('################################################\n')
# document results
conclusion = {'prob_thr':[thr1, thr2], 'precision':[precision, precision2], 'recall':[recall, recall2],
                    'f1':[f1, f12], 'accuracy':[accuracy, accuracy2], 
                    'roc_auc':[np.round(roc_auc_score(y_test, y_test_prob[:,1]), 2), 
                               np.round(roc_auc_score(y_test, y_test_prob2[:,1]), 2)]}
conclusion = pd.DataFrame(conclusion, index=[type(classifier).__name__, type(clf2).__name__])
print(np.round(conclusion,2).to_string())

# plot overlay ROC curves
plt.figure(figsize=[6,6])
plt.title('ROC curve comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
fpr, tpr, thrs = roc_curve(y_test, y_test_prob[:,1])
area_under_curve = auc(fpr, tpr)
plt.plot(fpr, tpr, label=type(classifier).__name__+' (AUC = %0.2f)' % area_under_curve)
fpr, tpr, thrs = roc_curve(y_test, y_test_prob2[:,1])
area_under_curve = auc(fpr, tpr)
plt.plot(fpr, tpr, label=type(clf2).__name__+' (AUC = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1], c='gray', linestyle='--', label='random classifier')
plt.legend(loc='lower right')
plt.show()

##########################################################################
# summary/conclusions
##########################################################################

# Two classifiers were used for a binary classification problem: to predict whether income exceeds $50K/yr using the Adult dataset.
# The column "class_>50K" was used as the expert label (corresponding to "class" column in the original dataset).
# A logistic regression model and a Naive Bayes model were used and assessed based on confusion matrix-derived metrics including: 
# precision, recall, and f1 score (which could provide more insight compared to accuracy for datasets with imbalanced classes such as this one),
# and based on ROC-based area under curve (AUC) metric. 
# For confusion matrix-derived metrics, the ROC curve was used to help pick a probability threshold. The general guideline was used:
# opting for the threshold that is based on the point in an ROC curve that is closest to the top-left corner.
# Note that a confusion matrix (and confusion matrix-derived metrics) are only valuable when its probability threshold can be specified and justfified.
# The ROC curve is the industry standard for classification accuracy, evaluting classifier performance for different threshold possibilities.

# For the logistic regression model:
# Based on the ROC curve and distance to the top-left corner, a probability threshold of 0.25 was used.
# From the resultant confusion matrix: precision=0.53, recall=0.85, F1=0.66 for outcome/class 1. (Overall accuracy=0.8.)
# From the ROC curve, AUC score=~0.9, which is a good result. (Per class handout, typical values range between 0.7 and 0.9)

# For the Naive Bayes model:
# Based on the ROC curve and distance to the top-left corner, a probability threshold of 0.95 was used.
# From the resultant confusion matrix: precision=0.44, recall=0.91, F1=0.60 for outcome/class 1. (Overall accuracy=0.72.)
# From the ROC curve, AUC score=~0.86, which is also a good result.

# Comparing the two models:
# ROC curves for the 2 models were compared.
# Intuitively, the curve that is closer to the top-left corner is the best. In this case, the logistic regression model performs better.
# Analytically, the curve with the largest area under it is the best. In this case, the logistic regression model performs better overall.
# Also, comparing confusion matrix-derived metrics at the selected thresholds, the logistic regression model has a higher F1 score.
