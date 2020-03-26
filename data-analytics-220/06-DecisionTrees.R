library(tidyverse)
library(GGally)
library(caret)
library(e1071)
library(gridExtra)

# 1. Read in the data using the read.delim function. 
# Then use ggplot2, ggpairs, and dplyr to identify interesting relationships in the data. 
# Write a short description of one interesting pattern you identified.

default = read.delim("credit_card_default.tsv", sep = "\t", header = TRUE)

# change categorical features and outcome variable to factors
default = default %>% mutate(default_next_month = as.factor(default_next_month),
                             sex = as.factor(sex),
                             education = as.factor(education),
                             marriage = as.factor(marriage))
levels(default$sex) = c('male', 'female')
levels(default$marriage) = c('others', 'married', 'single', 'divorced')
levels(default$education) = c('graduate', 'university', 'high', 'others')
levels(default$default_next_month) = c('no','yes')

# exploratory data analysis using ggplot2, ggpairs, and dplyr 
# explore data distributions and composition
summary(default$sex)
summary(default$education)
summary(default$marriage)
summary(default$age)

ggplot(default, aes(x = age)) + 
  geom_histogram() + 
  ggtitle('Histogram of age by sex') +
  facet_grid(~ sex)

ggplot(default, aes(x = education)) + 
  geom_bar() +
  ggtitle('data distribution') +
  facet_grid(~ sex)

ggplot(default, aes(x = marriage)) + 
  geom_bar() +
  ggtitle('data distribution') +
  facet_grid(~ sex)

# explore default status vs. features
ggplot(default, aes(x = default_next_month)) + 
  geom_bar() +
  ggtitle('default status by sex') +
  facet_grid(~ sex)
sum(default$sex=='male' & default$default_next_month=='yes')/sum(default$sex=='male')
sum(default$sex=='female' & default$default_next_month=='yes')/sum(default$sex=='female')
# Observation: males may be more likely to default (24% vs. 21% for females)

ggplot(default, aes(x = default_next_month)) + 
  geom_bar() +
  ggtitle('default status by marriage') +
  facet_grid(~ marriage)
sum(default$marriage=='married' & default$default_next_month=='yes')/sum(default$marriage=='married')
sum(default$marriage=='single' & default$default_next_month=='yes')/sum(default$marriage=='single')
# Observation: married people may be more likely to default (23% vs. 21% for singles)

ggplot(default, aes(x = default_next_month, y = limit_bal)) + 
  geom_boxplot() +
  ggtitle('limit_bal by default status')

ggplot(default, aes(x = default_next_month, y = limit_bal)) +
  geom_boxplot(aes(fill = default_next_month)) +
  facet_wrap(sex ~ marriage) +
  ggtitle('default status by sex and marriage')
# Observation: people who default generally have lower household credit limits across sex and marriage status.

# take a quick look at sex, marriage, limit balance, age using dplyr verbs
default %>% 
  group_by(sex, marriage) %>%
  summarize(n = n(), mean_limit_bal = mean(limit_bal), mean_age = mean(age))
# Observation: married male has the highest mean limit_bal, followed by married female and single female. 
# Divorced (male and female) tend to have the lowest limit_bal.

p1 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_sept + 1))) +
  geom_boxplot() 
p2 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_aug + 1))) +
  geom_boxplot() 
p3 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_july + 1))) + 
  geom_boxplot() 
p4 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_june + 1))) + 
  geom_boxplot() 
p5 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_may + 1))) + 
  geom_boxplot() 
p6 = ggplot(default, aes(x = default_next_month, y = log(pay_amt_april + 1))) + 
  geom_boxplot() 
grid.arrange(p1, p2, p3, p4, p5, p6, nrow = 2,
             top = "log of payment amount vs. default status")
# Observation: people who default generally have lower previous payment amount over the past 6 months.

ggpairs(select(default, limit_bal, bill_sept, bill_aug, bill_july, bill_june, bill_may, bill_april),
        title = 'Relationship between bill statement amounts and limit_bal')
# Obseration: high correlations between monthly bill amounts, but generally low correlations with limit_bal.

ggpairs(select(default, limit_bal, pay_amt_sept, pay_amt_aug, pay_amt_july, pay_amt_june, pay_amt_may, pay_amt_april),
        title = 'Relationship between payment amounts and limit_bal')
# Observation: low correlations between monthly payment amounts and with limit_bal.

# construct new feature sum_pay, calculated as the sum of all pay_xx columns. 
# Negative values (-1 and -2) are treated as 0, whereas positive values indicating payment delays are summed.
# A higher sum_pay indicates more/longer delayed payments in the past months, whereas 0 indicates no delays. 
temp=default[,7:12]
temp[temp<0]=0
default$sum_pay=rowSums(temp)

default %>% group_by(default_next_month) %>% 
  summarize(n = n(), mean_sum_pay = mean(sum_pay), median_sum_pay = median(sum_pay))

p1 = ggplot(default, aes(x = default_next_month, y = sum_pay)) +
  geom_boxplot(aes(fill = default_next_month)) +
  ggtitle('sum_pay by default status')

# Short description of one interesting pattern: 
# Defaulters generally have a pattern of delayed payment(s) in the previous months leading up to October
# This pattern is demonstrated by a higher mean and median sum_pay.

# construct new feature delay_sept indicating whether payment status in sept (last month) was late or not.
default$delay_sept = "delay"
default$delay_sept[default$pay_sept<0]="duly"

p2 = ggplot(default, aes(x = default_next_month)) +
  geom_bar(aes(fill = default_next_month)) +
  facet_wrap(~ delay_sept) +
  ggtitle('default status by pay_sept')

sum(default$default_next_month == "yes" & default$delay_sept == "delay")
sum(default$default_next_month == "no" & default$delay_sept == "delay")

sum(default$default_next_month == "no" & default$delay_sept == "duly")
sum(default$default_next_month == "yes" & default$delay_sept == "duly")

sum(default$default_next_month == "yes" & default$delay_sept == "duly")
sum(default$default_next_month == "yes" & default$delay_sept == "delay")

grid.arrange(p1, p2, nrow = 1)

ggpairs(select(default, default_next_month, sum_pay, delay_sept),
        mapping = aes(color = default_next_month, alpha = 0.5),
        title = 'Relationship between default next month, sum_pay, and pay_sept')

# Additional observed pattern:  
# People with delayed payment status in the month immediately prior (sept; pay_sept = delay) are more likely 
# to default next month (25% compared to 16% for people who paid duly in sept). 
# And of the defaulters, 80% had delayed payment status in the month immediately prior.

# 2. Construct at least one new feature to include in model development. 
# You might choose to create a new feature based on your findings from the exploratory data analysis. 
# Plot the new variable and interpret the result. 
# Use color to a facet to show the relationship between your new feature and the outcome variable default_next_month.

# Constructed 2 new features based on exploratory data analysis and plotted against default_next_month (above).

# end of exploratory data analysis, convert the engineered feature, along with payment status to factors
default = default %>% mutate(delay_sept = as.factor(delay_sept))

default = default %>% mutate(pay_april = as.factor(pay_april),
                             pay_may = as.factor(pay_may),
                             pay_june = as.factor(pay_june),
                             pay_july = as.factor(pay_july),
                             pay_aug = as.factor(pay_aug),
                             pay_sept = as.factor(pay_sept))

# 3. Use the createDataPartition function from the caret package to split the data into a training and testing set. 
# Pre-process the data with preProcess as needed.
set.seed(1)
in_train = createDataPartition(y = default$default_next_month, p = 0.8, list = FALSE)
default_train = default[in_train, ]
default_test = default[-in_train, ]

# select numerical variables, remove correlated variables, perform centering and scaling
preprocess_steps = preProcess(default_train %>% select_if(is.numeric), method = c("center", "scale", "corr"))
default_train_proc = predict(preprocess_steps, newdata = default_train)
default_test_proc = predict(preprocess_steps, newdata = default_test)

# remove variables for nzv
preprocess_steps2 = preProcess(default_train_proc, method = "nzv")
default_train_proc2 = predict(preprocess_steps2, newdata = default_train_proc)
# preprocess the test set using the means and sds from the training set
default_test_proc2 = predict(preprocess_steps2, newdata = default_test_proc)

str(default_train_proc2)

# 4. Fit at least 3 logistic regression models.
# logistic model
set.seed(1)
logistic_model = train(default_next_month ~ .,
                       data = default_train_proc2, 
                       method = "glm", 
                       family = binomial,
                       trControl = trainControl(method = "cv", number = 10))
summary(logistic_model)
logistic_predictions = predict(logistic_model, 
                               newdata = default_test_proc2)
confusionMatrix(logistic_predictions, default_test_proc2$default_next_month, positive = "yes")

# step model
# set.seed(1)
# step_model = train(default_next_month ~ .,
#                    data = default_train_proc2,
#                    method = "glmStepAIC",
#                    family = binomial,
#                    trControl = trainControl(method = "cv", number = 10))
# summary(step_model)
# step_predictions = predict(step_model,
#                            newdata = default_test_proc2)
# confusionMatrix(step_predictions, default_test_proc2$default_next_month, positive = "yes")

# bayesian model
set.seed(1)
bayesian_model = train(default_next_month ~ .,
                       data = default_train_proc2,
                       method = "bayesglm",
                       family = binomial,
                       trControl = trainControl(method = "cv", number = 10))
summary(bayesian_model)
bayesian_predictions = predict(bayesian_model, 
                               newdata = default_test_proc2)
confusionMatrix(bayesian_predictions, default_test_proc2$default_next_month, positive = "yes")

# boosted model
set.seed(1)
boosted_model = train(default_next_month ~ .,
                      data = default_train_proc2, 
                      method = "LogitBoost", 
                      family = binomial,
                      tuneLength = 10,
                      trControl = trainControl(method = "cv", number = 10))
summary(boosted_model)
boosted_predictions = predict(boosted_model, 
                              newdata = default_test_proc2)
confusionMatrix(boosted_predictions, default_test_proc2$default_next_month, positive = "yes")

# Use the defaults data set and the train function to build 3 decision trees to predict credit card default.
# tree model
set.seed(1)
tree_model = train(y = default_train_proc2$default_next_month, 
                   x = select(default_train_proc2, -default_next_month), 
                   method = "rpart",
                   tuneLength = 10,
                   trControl = trainControl(method = "cv", number = 10))
summary(tree_model)
tree_predictions = predict(tree_model, 
                           newdata = default_test_proc2)
confusionMatrix(tree_predictions, default_test_proc2$default_next_month, positive = "yes")

# bagged model
set.seed(1)
bagged_model = train(y = default_train_proc2$default_next_month, 
                   x = select(default_train_proc2, -default_next_month), 
                   method = "treebag",
                   trControl = trainControl(method = "cv", number = 10))
summary(bagged_model)
bagged_predictions = predict(bagged_model, 
                           newdata = default_test_proc2)
confusionMatrix(bagged_predictions, default_test_proc2$default_next_month, positive = "yes")

# ctree model
set.seed(1)
ctree_model = train(y = default_train_proc2$default_next_month, 
                     x = select(default_train_proc2, -default_next_month), 
                     method = "ctree",
                     tuneLength = 10,
                     trControl = trainControl(method = "cv", number = 10))
summary(ctree_model)
ctree_predictions = predict(ctree_model, 
                             newdata = default_test_proc2)
confusionMatrix(ctree_predictions, default_test_proc2$default_next_month, positive = "yes")

# 5. Use the dotplot function to compare the accuracy of the models you constructed in 4. 
# Which model performed the best in terms of predictive accuracy?

# compare models
results = resamples(list(logistic_model = logistic_model,
                         #step_model = step_model,
                         bayesian_model = bayesian_model,                         
                         boosted_model = boosted_model,
                         tree_model = tree_model,
                         bagged_model = bagged_model,
                         ctree_model = ctree_model))
summary(results)
dotplot(results)

ggplot(varImp(logistic_model)) + ggtitle('logistic model')
ggplot(varImp(bayesian_model)) + ggtitle('bayesian model')
ggplot(varImp(boosted_model)) + ggtitle('boosted model')
ggplot(varImp(tree_model)) + ggtitle('tree model')
ggplot(varImp(bagged_model)) + ggtitle('bagged model')
ggplot(varImp(ctree_model)) + ggtitle('ctree model')
