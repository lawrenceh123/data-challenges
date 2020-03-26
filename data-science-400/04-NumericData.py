"""
L04 Assignment: Numeric Data
Dataset: Credit Approval Data from UCI Machine Learning Repository
Operations: assign column names, impute and assign median values for missing numeric values,
replace outliers (with median values), create a histogram and a scatterplot, determine the standard deviation of all numeric variables
Please see summary comment block for discussion
"""

#1. Import statements for necessary package(s).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################

# 2. Read in the dataset from a freely and easily available source on the internet.
# using Credit Approval Data Set from UCI Machine Learning Repository
# this dataset contains: 
# --numeric attributes with missing data for this assignment
# --categorical attributes (also with missing data) for potential later use
# --specified attribute headings 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
Credit = pd.read_csv(url, header=None)

########################################

# 3. Assign reasonable column names.
# create column names as specified on dataset website (UCI Machine Learning Repository): A1, A2, ..., A16
# note that per dataset website: "all attribute names and values have been changed to meaningless symbols to protect confidentiality of the data."
Credit.columns = ['A{}'.format(i) for i in range(1,Credit.shape[1]+1)]

########################################

# 4. Impute and assign median values for missing numeric values.
# check data types
Credit.dtypes
# from dataset website, A2, A3, A8, A11, A14, A15 should be numeric,
# but A2 and A14 are have dtype "object", and have missing values ('?' as placeholders)

# Flag missing values in A2 and A14 and assign the median value
# A2
# convert to numeric data including nans
Credit.loc[:,'A2'] = pd.to_numeric(Credit.loc[:,'A2'], errors='coerce')
HasNan2 = np.isnan(Credit.loc[:,'A2'])
#sum(HasNan2) # num of missing values
Credit.loc[HasNan2, 'A2'] = np.nanmedian(Credit.loc[:,'A2'])

# A14
# convert to numeric data including nans
Credit.loc[:,'A14'] = pd.to_numeric(Credit.loc[:,'A14'], errors='coerce')
HasNan14 = np.isnan(Credit.loc[:,'A14'])
#sum(HasNan14) # num of missing values
Credit.loc[HasNan14, 'A14'] = np.nanmedian(Credit.loc[:,'A14'])

# Create histograms + scatterplots for all numeric variables before replacing outliers
# histograms can be used to visualize outliers for the next step
pd.plotting.scatter_matrix(Credit, alpha=0.2, figsize=[10,10]) 
plt.suptitle('Histogram and scatterplots of numeric variables before replacing outliers')
plt.show()

########################################

#5. Replace outliers.
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
Credit[['A2', 'A3', 'A8', 'A11', 'A14', 'A15']] = Credit[['A2', 'A3', 'A8', 'A11', 'A14', 'A15']].apply(replace_outlier)

########################################

# 6. Create a histogram of a numeric variable. Use plt.show() after each histogram.
# create a histogram of numeric variable A2
plt.hist(Credit.loc[:, 'A2']) 
plt.xlabel('A2')
plt.ylabel('Count')
plt.title('Histogram of A2 after replacing outliers')
plt.show()

# 7. Create a scatterplot. Use plt.show() after the scatterplot.
# create a scatterplot of A3 vs. A2
plt.scatter(Credit.loc[:,'A2'], Credit.loc[:,'A3'])
plt.xlabel('A2')
plt.ylabel('A3')
plt.title('Scatterplot of A3 vs. A2 after replacing outliers')
plt.show()

# from assginment instructions, the number of histograms and scatterplots requested was unclear. (one numeric variable? all numeric variables?)
# Create histograms + scatterplots for all numeric variables 
pd.plotting.scatter_matrix(Credit, alpha=0.2, figsize=[10,10]) 
plt.suptitle('Histogram and scatterplots of numeric variables after replacing outliers')
plt.show()

########################################

# 8. Determine the standard deviation of all numeric variables. Use print() for each standard deviation.
# std for all numeric variables
stds = np.std(Credit)
# per instruction, use print() for each std
for i, x in enumerate(stds):
    print('Numeric variable:', stds.index[i], ', Standard deviation:', x)

########################################
    
# 9. Add comments to explain the code blocks.
# Please see comments in code blocks above.

########################################

# 10. Add a summary comment block on how the numeric variables have been treated: 

# 1. Which attributes had outliers and how were the outliers defined? 
# ANS: Referencing the histograms (after median imputation of missing values but before replacing outliers),
# the standard definition of outliers was used in the absence of additional domain knowledge:
# x < mean(x)-2*std(x) or x > mean(x)+2*std(x).
# By this definition, numeric attributes A2, A3, A8, A11, A14, A15 had outliers.

# 2. Which attributes required imputation of missing values and why? 
# ANS: Numeric attributes A2 and A14 had missing values with '?' as placeholders and required imputation.
# Missing values in numeric attributes make the column elements as datatype object and represented as strings,  
# which prevents math operations such as calcuating the standard deviation, or plotting a (correct) histogram.
# In addition, subsequent analyses may not tolerate the placeholders.
   
# 3. Which attributes were histogrammed and why?  
# ANS: histograms were plotted for all numerical attributes to examine their distributions and visualize outliers.
       
# 4. Which attributes were removed and why?  
# ANS: No numeric attribtues were removed. Per assignment instructions, median imputation of missing values was performed instead.
# The advantage of this replacement option is that data are not lost. 
# The disadvantage is that the replacement value is based on a guess (median in this case).
# If numeric attribtues with missing data was to be removed instead, then all attributes/columns that have one or more NaN should be removed.
# (Missing categorical values were not replaced or removed in this assignment.)
    
# 5. How did you determine which rows should be removed?
# ANS: If missing data was to be removed using the row removal method, then all rows that have one or more NaN should be removed.
# However, per assignment instructions, median imputation of missing values was performed for numeric attributes, 
# and no rows were removed. (Per assignment instructions, outliers were also replaced and not removed.)    

# The following could be used to remove attributes or rows with missing values:
## After reading in the dataset, Replace '?' with NaNs
# Credit = Credit.replace(to_replace="?", value=float("NaN"))
## Remove columns that contain one or more NaN
# Credit_FewerCols = Credit.dropna(axis=1)
## Remove rows that contain one or more NaN
# Credit_FewerRows = Credit.dropna(axis=0)

########################################
