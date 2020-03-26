"""
Lesson 8 Assignment: Predictive Analytics
1. Short narrative on the data preparation for your chosen data set from Milestone 2.
2. Import statements for necessary package(s).
3. Read in the dataset from a freely and easily available source on the internet.
4. Show data preparation. Normalize some numeric columns, one-hot encode some categorical columns with 3 or more categories, remove or replace missing values, remove or replace some outliers.
5. Ask a binary-choice question that describes your classification. Write the question as a comment. Specify an appropriate column as your expert label for a classification (include decision comments).
6. Apply K-Means on some of your columns, but make sure you do not use the expert label. Add the K-Means cluster labels to your dataset.
7. Split your data set into training and testing sets using the proper function in sklearn (include decision comments).
8. Create a classification model for the expert label based on the training data (include decision comments).
9. Apply your (trained) classifiers to the test data to predict probabilities.
10. Write out to a csv a dataframe of the test data, including actual outcomes, and the probabilities of your classification.
11. Determine accuracy rate, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary).
12. Add comments to explain the code blocks.
13. Add a summary comment block that discusses your classification.
"""

# Note: Step 1 (short narrative on data preparation) follows Step 4
##########################################################################
# 2. Import statements for necessary package(s).
##########################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

##########################################################################
# 3. Read in the dataset from a freely and easily available source on the internet.
# 4. Show data preparation. Normalize some numeric columns, one-hot encode some categorical columns with 3 or more categories, 
# remove or replace missing values, remove or replace some outliers.
##########################################################################

# Import cleaned up data, using code developed for Milestone 2
def prepareAdultData():
    print('Begin data import and preparation...')
    print('#############################')      
    # 1. Import statements for necessary package(s).
    #import numpy as np
    #import pandas as pd
    #from sklearn.preprocessing import MinMaxScaler
    
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
    Adult_original = Adult.copy()
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
    
    # 7. Consolidate categorical data (at least 1 column).
    
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
    
    # simplify occupation by consolidating [Other-service, Tech-support, Protective-serv, 
    # Priv-house-serv, Armed-Forces] as Service
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
    
    ## 10. Present plots for 1 or 2 categorical columns.
    ## create plots before performing one-hot-encoding and removing obsolete columns
    ## step 8, 9 will follow below
    #
    ## categorical column: education (consolidated; 16 categories into 5)
    #Adult['education-c'].value_counts().plot(kind='bar')
    #plt.title('categorical column: education (consolidated)')
    #plt.ylabel('count')
    #plt.xticks(rotation=0)
    #plt.show()
    #
    ## categorical column: martial-status (consolidated; 7 categories into 3)
    #Adult['martial-status-c'].value_counts().plot(kind='bar')
    #plt.title('categorical column: martial-status (consolidated)')
    #plt.ylabel('count')
    #plt.xticks(rotation=0)
    #plt.show()
    
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
    
    # 8. One-hot encode categorical data 
    
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
# 1. Short narrative on the data preparation for your chosen data set from Milestone 2.
##########################################################################

# -number of observations and attributes
# observations: 32561
# attributes: 12 original attributes, 
# expanded to 32 after one-hot encoding of categorical variables (with all categories except for one)
Adult_df.info()
# -datatype, distribution, and a comment on each attribute
# attributes:
# 1. age: numeric, see histogram for distribution, represents age 
# outliers (2*std) removed and min-max normalized
Adult_df['age'].describe()
Adult_df['age'].hist() # most values around 0.5 after min-max normalization
plt.ylabel('count')
plt.title('histogram of age')
plt.show()
# 2. hours-per-week: numeric, see histogram for distribution, represents hours worked per week
# outliers (2*std) removed and min-max normalized
Adult_df['hours-per-week'].describe()
Adult_df['hours-per-week'].hist() # most values around 0.5 after min-max normalization
plt.ylabel('count')
plt.title('histogram of hours-per-week')
plt.show()
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
# 5. Ask a binary-choice question that describes your classification. 
# Write the question as a comment. 
# Specify an appropriate column as your expert label for a classification (include decision comments).
##########################################################################

# Binary-choice question: Does income exceed $50k per year? 
# Expert label column: class_>50K
# Decision comment: class_>50K is binary, with 1 and 0 representing income >50K and <=50K a year, respectively,
# and is appropriate for this question.

##########################################################################
# 6. Apply K-Means on some of your columns, but make sure you do not use the expert label. 
# Add the K-Means cluster labels to your dataset.
##########################################################################

# X: new copy of data with the selected attributes for clustering
X = Adult_df.copy()[['age', 'hours-per-week', 'workclass_NoPay', 'workclass_Private', 'workclass_SelfEmp',
                 'sex_Male']]

# estimate optimal number of clusters; compare within-cluster sum of squares for k=2 to k=7
wcss = []
for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(2,8), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.title('find optimal number of clusters')
plt.show()

# pick k = 3 based on the elbow method (4 could also work but will try 3 here)
kmeans = KMeans(n_clusters=3)
Y = kmeans.fit_predict(X)

# Add the cluster label to the dataset.
Adult_df['cluster'] = Y
# to be consistent with other columns, one-hot encode cluster label, 
# represent all categories except for one to avoid a linearly dependent data set
Adult_df = pd.get_dummies(Adult_df, prefix='cluster', columns=['cluster'], drop_first=True)

##########################################################################
# 7. Split your data set into training and testing sets using the proper function in sklearn 
# (include decision comments).
##########################################################################

X = Adult_df.copy().drop('class_>50K', axis=1) #features
y = Adult_df.copy()['class_>50K'] #expert label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Decision comment: use most of the data for training. A 80/20 train/test split is often reasonable.

##########################################################################
# 8. Create a classification model for the expert label based on the training data 
# (include decision comments).
##########################################################################

# train a logistic regression classficiation model
classifier = LogisticRegression(solver='saga') 
classifier.fit(X_train, y_train) 
# Decision comment: a logistic regression classifier is well-suited for binary classification problems,
# returning probability estimates that could be used to predict class labels.

##########################################################################
# 9. Apply your (trained) classifiers to the test data to predict probabilities.
##########################################################################

y_test_prob = classifier.predict_proba(X_test) # probability estimates
y_test_pred = classifier.predict(X_test) # predict class labels
# The threshold for probability of classification was 0.5. 
# (e.g. If P(Y=0)>0.5 then it was assigned label 0.)

##########################################################################
# 10. Write out to a csv a dataframe of the test data, including actual outcomes, 
# and the probabilities of your classification.
##########################################################################

test_df = X_test.copy() # df of test data to be written 
test_df['class_>50K'] = y_test # actual outcome/expert label (1: >50K, 0:<=50K)
test_df['class_prob0']= y_test_prob[:,0] # prob of classification for outcome 0
test_df['class_prob1']= y_test_prob[:,1] # prob of classification for outcome 1
test_df['class_pred']=y_test_pred # predicted outcome
test_df.to_csv('test_data.csv', index=False) # write to csv

##########################################################################
# 11. Determine accuracy rate, which is the number of correct predictions divided by 
# the total number of predictions (include brief preliminary analysis commentary).
##########################################################################

# compute accuracy: number of correct predictions divided by the total number of predictions
print('test set accuracy, calculated:',sum(y_test_pred==y_test)/len(y_test))
# get accuracy using score method (same result as expected)
print('check accuracy using sklearn:',classifier.score(X_test, y_test))
# Preliminary analysis comment: the overall accuracy of 0.85 is a reasonable result.
# However, the classes are imbalanced (>50K: ~23%, <=50K: ~77%), and additional metrics
# (precision, recall) are needed to give more insight.
print(classification_report(y_test, y_test_pred))

##########################################################################
# 12. Add comments to explain the code blocks.
##########################################################################

# please see above code blocks

##########################################################################
# 13. Add a summary comment block that discusses your classification.
##########################################################################

# Summary: the Adult dataset was prepared for analysis in previous assignments.
# K-Means clustering was applied using selected columns, and the cluster labels were added to the data.
# Data was split into 80% training set and 20% test set.
# A logistic regression classifier was used for a binary classification problem: 
# to predict whether income exceeds $50K/yr. The column class_>50K was used as the expert label 
# (corresponding to "class" column in the original dataset).
# Preliminary analysis returned an overall accuracy of 0.85, a resonable result.
# However, since the classes are imbalanced, additional metrics are needed for more insight.
