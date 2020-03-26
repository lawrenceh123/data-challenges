"""
L03 Assignment: Aberrant Data
Clean data by removing or replacing outliers, or by replacing improper/missing values
"""

# 1. Import necessary packages
import numpy as np

# 2. Create a numeric numpy array, named arr1, with at least 30 items that contains outliers
nums=[77, 1, 8, 4, 9, 1, 6, 3, 7, 1, 4, 3, 9, 7, -88, 5, 6, 7, 3, 2, 6, 2, 8, 7, 6, 3, 2, 7, 99, 3]
arr1 = np.array(nums) # numeric numpy array

# 3. Create a numpy array that should have been numeric, named arr2. arr2 contains improper non-numeric missing values, like "?"
nums[0]="?"
nums[3]="???"
nums[14]=" "
nums[28]=""
arr2 = np.array(nums) #numpy array containing numeric and improper non-numeric values

########################################

# 4. Create (define) a function, named "remove_outlier", that removes outliers in arr1
def remove_outlier(x):
    # Limits for acceptable values: mean+2*std
    LimitHi = np.mean(x) + 2*np.std(x)
    LimitLo = np.mean(x) - 2*np.std(x)
    # Create Flag for acceptable values (non-outliers)
    FlagGood = (x >= LimitLo) & (x <= LimitHi)
    # Return non-outliers
    return x[FlagGood]

# 5. Create (define) a function, named "replace_outlier", that replaces outliers in arr1 with the arithmetic mean of the non-outliers
def replace_outlier(y):
    x = np.copy(y)
    # Limits for acceptable values: mean+2*std
    LimitHi = np.mean(x) + 2*np.std(x)
    LimitLo = np.mean(x) - 2*np.std(x)
    # Create Flags for outliers and non-outliers
    FlagBad = (x < LimitLo) | (x > LimitHi)
    FlagGood = ~FlagBad
    # Replace outliers with the mean of non-outliers
    x = x.astype(float) # since np.mean returns float
    x[FlagBad] = np.mean(x[FlagGood])
    return x

# 6. Create (define) a function, named "fill_median", that fills in the missing values in arr2 with the median of arr2
def fill_median(y):
    x = np.copy(y)
    # Find elements that are numbers
    FlagGood = [element.isdigit() for element in x]
    # Find elements that are improper missing values
    FlagBad = [not element.isdigit() for element in x]
    # Replace improper missing values with the median of the numeric values
    x[FlagBad] = np.median(x[FlagGood].astype(int))
    return x.astype(float) # use float to account for non-numeric values with length >1 e.g. "???" or "   "

########################################

# 7. Call the three functions with their appropriate arrays in your script
arr1_remove = remove_outlier(arr1)
arr1_replace = replace_outlier(arr1)
arr2_replace = fill_median(arr2)

########################################

# 8. Comments explaining the code blocks
# See comments with each code block

########################################

# 9. Summary comment block on how your dataset has been cleaned up

# Summary:
# Using remove_outlier, outliers (outside of 2*std of the mean) in arr1 have been removed.
# Using replace_outlier, outliers in arr1 have been replaced with the mean of non-outliers.
# Using fill_median, non-numeric missing values in arr2 have been replaced with the median of numeric values in arr2.

########################################
