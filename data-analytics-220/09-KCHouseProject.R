# load libraries
if(!require("randomForest")) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require("corrgram")) install.packages("corrgram", repos = "http://cran.us.r-project.org")
if(!require("caret")) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require("reshape2")) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require("GGally")) install.packages("reshape2", repos = "http://cran.us.r-project.org")

library(corrgram)
library(randomForest)
library(GGally)
library(caret)
library(tidyverse)
library(ggplot2)
library(reshape2)

# load file and prepare data
house = read_csv("kc_house_data.csv")
#glimpse(house)

# features with named categories or with a small number of discrete numeric values
# were considered categorical features.
house = house %>% mutate(waterfront_fac = as.factor(waterfront), 
                         view_fac = as.factor(view), 
                         condition_fac = as.factor(condition), 
                         grade_fac = as.factor(grade), 
                         zipcode_fac = as.factor(zipcode))
glimpse(house)
###########################################

# define plot_hist to plot histograms
plot_hist = function(col, df, bins = 20){
  p1 = ggplot(df, aes_string(col)) + 
    geom_histogram(aes(y = ..density..), bins = bins, 
                   alpha = 0.3, color = 'blue') +
    geom_density(size = 1) +
    xlab(col) +
    ggtitle(paste('Histogram and density function for', col))
  print(p1)
}

plot_hist('price', house)
# The distribution of price is positively-skewed with a long right tail
house = house %>% mutate(price_log = log(price))
plot_hist('price_log', house)
# A log transformation creates a distribution closer to Normal

# variable selection
# 1. Evaluate variable correlations
# 2. Remove highly correlated independent variables. 
# Recall the hint about sqft_living15.
# 3. Evaluate variable significance.
# For the final selection consider variables with the following:
#   be important for your decision (those may have low importance)
#   don't impact model overfitting
#   don't impact performance of the algorithm (keep in mind that each additional variable slows model training)
# 4. Remove variables with low significance

# id is a transaction id that is independent of price; remove id
house$id = NULL
# following the provided hint, compare sqft_living15 and sqft_lot15 to other features of similar names.
cor(select(house, sqft_living, sqft_living15))
cor(select(house, sqft_lot, sqft_lot15))
# remove sqft_living15 and sqft_lot15 
house$sqft_living15 = NULL
house$sqft_lot15 = NULL

ggpairs(house[,c(3:7,24)]) # price_log
# display price plus next set of columns from house that reasonably fit on the screen 
ggpairs(house[,c(12:15,17:18,24)]) # price_log
ggpairs(house[,c(19:22,24)]) # price_log

ggplot(house, aes(x = zipcode_fac, y = price_log)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75),
              fill = 'blue', alpha = 0.3, size = 1.0) +
  xlab('zipcode_fac') + ylab('price_log') +
  ggtitle('price_log vs. zipcode_fac')

# sqft_living is also skewed
house = house %>% mutate(sqft_living_log = log(sqft_living))
ggpairs(select(house, sqft_living, sqft_living_log, price_log))
# A log transformation creates a distribution closer to Normal
# Also, relationship between sqft_living_log and price_log is more linear

# top correlations
cor_level = 0.7
#correlationMatrix = cor(house %>% select_if(is.numeric))
#correlationMatrix = cor(house %>% select_if(is.numeric) %>% select(-price))
correlationMatrix = cor(house %>% select_if(is.numeric) %>% select(-price, -sqft_living))
cor_melt = arrange(melt(correlationMatrix),desc(value))

#hint: you can use %% 2 to remove every other row so we don't get duplicates, given 
#hint: several correlated variables interract with each other given us duplicates from melt
dplyr::filter(cor_melt, dplyr::row_number() %% 2 == 0, value >= cor_level & value != 1)

#show variables that correlate to price only
cor_level = 0.5
dplyr::filter(cor_melt, dplyr::row_number() %% 2 == 0, value >= cor_level & value != 1, Var1 == 'price_log' | Var2 == 'price_log')

#show variables have low correlation to price
dplyr::filter(cor_melt, dplyr::row_number() %% 2 == 0, value != 1, Var1 == 'price_log' | Var2 == 'price_log') %>% tail()

# remove price: use log_price
# remove zipcode_fac: randomForest can not handle categorical predictors with more than 53 categories
#house2 = house %>% select(-price, -waterfront, -view, -condition, -grade, -zipcode, -zipcode_fac)
#house2 = house %>% select(-price, -waterfront, -view, -condition, -grade, -zipcode, -zipcode_fac, -sqft_above)
house2 = house %>% select(-price, -waterfront, -view, -condition, -grade, -zipcode, -zipcode_fac, -sqft_living, -sqft_above)

comboInfo <- findLinearCombos(house2 %>% select_if(is.numeric) %>% select(-price_log))
#no linear combinations

# split the data into a training and testing set
set.seed(123)
in_train = createDataPartition(y = house2$price_log, p = 0.8, list = FALSE)
house_train = house2[in_train, ]
house_test = house2[-in_train, ]

# preprocess: center, scale, nzv
#preprocess_steps = preProcess(house_train %>% select_if(is.numeric) %>% select(-price_log), method = c("center", "scale"))
preprocess_steps = preProcess(house_train %>% select(-price_log), method = c("center", "scale", "nzv"))
house_train_proc = predict(preprocess_steps, newdata = house_train)
house_test_proc = predict(preprocess_steps, newdata = house_test)
preprocess_steps
preprocess_steps$method

glimpse(house_train_proc)
###########################################

#evaluate models for initial performance
#tree model/random forest
rf_model_test = randomForest(price_log ~ ., data = house_train_proc)
summary(rf_model_test)
plot(rf_model_test)

var_imp = importance(rf_model_test)
var_imp
varImpPlot(rf_model_test)
rf_model_test

#linear model
lm_model_test = lm(price_log ~ ., data = house_train_proc)
summary(lm_model_test)
confint(lm_model_test)

#linear model2 (remove long--coefficient not significant)
lm_model_test2 = lm(price_log ~ date + bedrooms + bathrooms + sqft_lot + floors +
                     yr_built + lat + waterfront_fac + view_fac + condition_fac +
                     grade_fac + sqft_living_log, 
                   data = house_train_proc)
summary(lm_model_test2)
confint(lm_model_test2)

###########################################

# using train function
# linear model
set.seed(123)
train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
linear_model <- train(price_log ~ date + bedrooms + bathrooms + sqft_lot + floors +
                        yr_built + lat + waterfront_fac + view_fac + condition_fac +
                        grade_fac + sqft_living_log,
                   data = house_train_proc,
                   trControl = train_control,
                   method = "lm")
linear_model
ggplot(varImp(linear_model)) + ggtitle('linear model')

# tree model
set.seed(123)
train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
tree_model <- train(price_log ~ .,
                  data = house_train_proc,
                  trControl = train_control,
                  tuneLength = 10,
                  method = "rpart")
ggplot(varImp(tree_model)) + ggtitle('tree model')

# random forest 
set.seed(123)
train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
rf_model <- train(price_log ~ .,
                  data = house_train_proc,
                  trControl = train_control,
                  tuneLength = 10,
                  importance = TRUE,
                  method = "rf")
rf_model
ggplot(varImp(rf_model)) + ggtitle('random forest')
#plot(rf_model$finalModel)
###########################################

# compare models
results = resamples(list(linear_model = linear_model,
                         tree_model = tree_model,
                         random_forest = rf_model))
summary(results)
dotplot(results)

# make prediction using rf on house_test_proc
rf_pred = predict(rf_model, newdata = house_test_proc) 
rf_pred_err = postResample(pred = rf_pred,
                           obs = house_test_proc$price_log)

# plot for selected final model
errors = data.frame(predicted = rf_pred %>% exp(), 
                    observed = house_test_proc$price_log %>% exp(), 
                    error = rf_pred %>% exp() - house_test_proc$price_log %>% exp())
ggplot(data = errors, aes(x = observed, y = predicted)) +
  geom_point(alpha = 0.25) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  geom_smooth(method = "lm") +
  ggtitle('Observed vs. predicted price using random forest model') +
  coord_equal()



##################################
# prediction of home price
# using the following 12 features as instructed
# use all available data as train set and to_predict defined as test set
# house3_train = house %>% select(price_log, bedrooms, bathrooms,
#                                 sqft_living, sqft_lot, floors,
#                                 waterfront, view, condition, grade, sqft_above)
house3_train = house %>% select(price_log, bedrooms, bathrooms,
                                sqft_living, sqft_lot, floors,
                                waterfront_fac, view_fac, condition_fac, grade_fac, sqft_above)
house3_train = house3_train %>% mutate(sqft_living_sqr = sqrt(sqft_living),
                                       sqft_living_log = log(sqft_living))

#predict for grade 7 and for the maximum grade (13)
#condition of 5 is already max in dataset
to_predict = house3_train[0,]
to_predict[1,]$price_log = 0
to_predict[1,]$bedrooms = 4
to_predict[1,]$bathrooms = 3
to_predict[1,]$sqft_living = 4000
to_predict[1,]$sqft_lot = 5000
to_predict[1,]$floors = 1
# to_predict[1,]$waterfront = 0
# to_predict[1,]$view = 0
# to_predict[1,]$condition = 5
# to_predict[1,]$grade = 7
to_predict[1,]$waterfront_fac = 0
to_predict[1,]$view_fac = 0
to_predict[1,]$condition_fac = 5
to_predict[1,]$grade_fac = 7
to_predict[1,]$sqft_above = 4000
to_predict[1,]$sqft_living_sqr = sqrt(4000)
to_predict[1,]$sqft_living_log = log(4000)
#to_predict[1,]$yr_built = 2004
# note: yr_built not in list of features in requested in instructions
to_predict[2,]$price_log = 0
to_predict[2,]$bedrooms = 4
to_predict[2,]$bathrooms = 3
to_predict[2,]$sqft_living = 4000
to_predict[2,]$sqft_lot = 5000
to_predict[2,]$floors = 1
# to_predict[2,]$waterfront = 0
# to_predict[2,]$view = 0
# to_predict[2,]$condition = 5
# to_predict[2,]$grade = 13
to_predict[2,]$waterfront_fac = 0
to_predict[2,]$view_fac = 0
to_predict[2,]$condition_fac = 5
to_predict[2,]$grade_fac = 13
to_predict[2,]$sqft_above = 4000
to_predict[2,]$sqft_living_sqr = sqrt(4000)
to_predict[2,]$sqft_living_log = log(4000)
#to_predict[2,]$yr_built = 2004

#summary(to_predict)
house3_test = to_predict

# scale features 
# cols = c("bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_living_sqr", "sqft_living_log")
# house3_train[, cols] = lapply(house3_train[, cols], scale)
# would like to center and scale the test data (to_predict) based on the train set. 
# could use the following:
#aaa=lapply(house3_train[, cols], scale)
#(house3_test$bedrooms-attr(aaa$bedrooms, "scaled:center"))/attr(aaa$bedrooms, "scaled:scale")
#(house3_test$bathrooms-attr(aaa$bathrooms, "scaled:center"))/attr(aaa$bathrooms, "scaled:scale")
# but easier with caret

# center and scale
preprocess_steps = preProcess(house3_train %>% select_if(is.numeric) %>% select(-price_log), method = c("center", "scale"))
#preprocess_steps = preProcess(house3_train %>% select(-price_log), method = c("center", "scale"))
house3_train_proc = predict(preprocess_steps, newdata = house3_train)
house3_test_proc = predict(preprocess_steps, newdata = house3_test)
preprocess_steps
preprocess_steps$method

# use train
train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)

set.seed(123)
p2linear_model <- train(price_log ~ .,
                  data = house3_train_proc,
                  trControl = train_control,
                  method = "lm")
summary(p2linear_model)

# remove non-significant variable (sqft_lot) and train again
set.seed(123)
p2linear_model2 <- train(price_log ~ bedrooms + bathrooms + sqft_living + floors +
                           waterfront_fac + view_fac + condition_fac + grade_fac +
                           sqft_above + sqft_living_sqr + sqft_living_log,
                         data = house3_train_proc,
                         trControl = train_control,
                         method = "lm")
summary(p2linear_model2)
pred_price_lm = predict(p2linear_model2, newdata = house3_test_proc) %>% exp()

# use lm to get confint
set.seed(123)
p2lm_model_test = lm(price_log ~ bedrooms + bathrooms + sqft_living + floors +
                       waterfront_fac + view_fac + condition_fac + grade_fac +
                       sqft_above + sqft_living_sqr + sqft_living_log,
                     data = house3_train_proc)
summary(p2lm_model_test)
confint(p2lm_model_test)

# random forest
set.seed(123)
p2rf_model <- train(price_log ~ .,
                    data = house3_train_proc,
                    trControl = train_control,
                    importance = TRUE,
                    method = "rf")
summary(p2rf_model)
pred_price_rf = predict(p2rf_model, newdata = house3_test_proc) %>% exp()

# compare models
results = resamples(list(linear_regression = p2linear_model2,
                         random_forest = p2rf_model))
summary(results)
dotplot(results)
