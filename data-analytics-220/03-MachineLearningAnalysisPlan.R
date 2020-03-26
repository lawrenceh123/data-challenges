library(tidyverse)
library(caret)
library(GGally)

set.seed(1234)

ff = read.delim('forestfires.tsv', header = TRUE, sep = '\t')

# add a log transformed version of area to use as the outcome
ff$log_area = log10(ff$area + 1)

#1. Make at least one new feature and plot it against the burn area. 
#From visual inspection does there appear to be a relationship between the new feature and the burn area?

# new variable: is_summer
ff$is_summer = factor(ifelse(ff$month %in% c('jun', 'jul', 'aug'), 1, 0))
ggpairs(select(ff, temp, RH, wind, rain, log_area, is_summer) %>%
          filter(log_area > 0), mapping = aes(color = is_summer, alpha = 0.5), title = 'Visual inspection of is_summer vs. log_area and meterologoical features')
# No, by visual inspection is_summer does not appear to affect burn area (for log_area > 0;
# although the median log_area for is_summer is slightly decreased compared to not is_summer).
# The visualization does not support the hypothesis that the summer climate (mainly higher temp as judged from ggpairs)
# makes it more likely that there will be fires.

#new variable: is _weekend
ff$is_weekend = factor(ifelse(ff$day %in% c("sat", "sun"), 1, 0))
ggpairs(select(ff, temp, RH, wind, rain, log_area, is_weekend) %>%
          filter(log_area > 0), mapping = aes(color = is_weekend, alpha = 0.5), title = 'Visual inspection of is_weekend vs. log_area and meterologoical features')
# No, by visual inspection, is_weekend does not appear to affect burn area (for log_area > 0).
# The visualization does not support the hypothesis that more visitors on the weekend makes it more likely that there
# will be fires.

# convert month and day into a series of binary variables
ff$month = factor(ff$month, levels = c('jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
ff$day = factor(ff$day, levels = c('mon', 'tue', 'wed', 'thu','fri', 'sat', 'sun'))
month = model.matrix(~month - 1, data = ff) 
day = model.matrix(~day - 1, data = ff)
ff = cbind(ff, month, day) 
ff = select(ff, -month, -day)

# 2. Use createDataPartition to split 80% of the forest fire data into a training set.
in_train = createDataPartition(y = ff$log_area, p = 0.8, list = FALSE)
#training set
ff_train = ff[in_train, ]
#test set
ff_test = ff[-in_train, ]

# 3. Use preProcess to prepare your data for analysis. 
#What, if any, variables were removed for near zero variance?

# select numerical variables for centering and scaling
preprocess_steps = preProcess(select(ff_train, FFMC, DMC, DC, ISI, temp, RH, wind, rain), method = c("center", "scale"))

ff_train_proc = predict(preprocess_steps, newdata = ff_train)
# preprocess the test set using the means and sds from the training set
ff_test_proc = predict(preprocess_steps, newdata = ff_test)

# remove variables for nzv
preprocess_steps2 = preProcess(ff_train_proc, method = "nzv")
ff_train_proc2 = predict(preprocess_steps2, newdata = ff_train_proc)
# preprocess the test set using the means and sds from the training set
ff_test_proc2 = predict(preprocess_steps2, newdata = ff_test_proc)

setdiff(names(ff_train_proc),names(ff_train_proc2))
# variables that were removed for near zero variance are:
# rain, monthjan, monthfeb, monthapr, monthmay, monthjun, monthoct, monthnov, monthdec

