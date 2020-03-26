library(tidyverse)
library(caret)
library(GGally)

ff = read.delim('forestfires.tsv', header = TRUE, sep = '\t')

ggplot(ff, aes(x = area)) + geom_histogram() + ggtitle('Histogram of area')
# what fraction of area observations are zeros?
table(ff$area==0)/nrow(ff)

# add a log transformed version of area to use as the outcome variatble
ff$log_area = log10(ff$area + 1)
ggplot(ff, aes(x = log_area)) + geom_histogram() + ggtitle('Histogram of log10(area + 1)')
ggplot(filter(ff, log_area > 0), aes(x = log_area)) + geom_histogram() + ggtitle('Histogram of log10(area + 1) for log10(area + 1) > 0')

# create new features
ff$is_summer = factor(ifelse(ff$month %in% c('jun', 'jul', 'aug'), 1, 0))
ff$is_weekend = factor(ifelse(ff$day %in% c("sat", "sun"), 1, 0))

# convert X, Y, month, and day into a series of binary variables
ff$month = factor(ff$month, levels = c('jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
ff$day = factor(ff$day, levels = c('mon', 'tue', 'wed', 'thu','fri', 'sat', 'sun'))
month = model.matrix(~month - 1, data = ff) 
day = model.matrix(~day - 1, data = ff)
ff$X = factor(ff$X)
ff$Y = factor(ff$Y)
x = model.matrix(~X - 1, data = ff)
y = model.matrix(~Y - 1, data = ff)
ff = cbind(ff, month, day, x, y) 
ff = select(ff, -X, -Y, -month, -day, -area)

set.seed(1234)
# split 80% of the forest fire data into a training set.
in_train = createDataPartition(y = ff$log_area, p = 0.8, list = FALSE)
#training set
ff_train = ff[in_train, ]
#test set
ff_test = ff[-in_train, ]

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

# identify correlated predictors with correlations above 0.75
ffcor = cor(select(ff_train, FFMC, DMC, DC, ISI, temp, RH, wind, rain))
summary(ffcor[upper.tri(ffcor)])
highcor = findCorrelation(ffcor, cutoff = 0.75)

# Regression with train function
# full model
set.seed(1234)
full_model = train(log_area ~ .,
                   data = ff_train_proc2,
                   method = "lm",
                   trControl = trainControl(method = "cv", number = 10))

full_model_pred = predict(full_model, newdata = ff_test_proc2)
full_model_pred_err = postResample(pred = full_model_pred, 
                                   obs = ff_test_proc2$log_area)
summary(full_model)

#forward model
set.seed(1234)
forward_model = train(log_area ~ ., 
                      data = ff_train_proc2, 
                      method = "leapForward",
                      tuneGrid = expand.grid(nvmax = 1:20),
                      #tuneLength = 20,
                      trControl = trainControl(method = "cv", number = 10))
ggplot(forward_model) + ggtitle('Forward model: nvmax = 1:20')
ggplot(varImp(forward_model)) + ggtitle('Feature importance in forward model')

forward_model_pred = predict(forward_model, newdata = ff_test_proc2)
forward_model_pred_err = postResample(pred = forward_model_pred,
                                      obs = ff_test_proc2$log_area)
summary(forward_model)

#backward model
set.seed(1234)
backward_model = train(log_area ~ ., 
                      data = ff_train_proc2, 
                      method = "leapBackward",
                      tuneGrid = expand.grid(nvmax = 1:20),
                      trControl = trainControl(method = "cv", number = 10))
ggplot(backward_model) + ggtitle('Backward model: nvmax = 1:20')
ggplot(varImp(backward_model)) + ggtitle('Feature importance in backward model')

backward_model_pred = predict(backward_model, newdata = ff_test_proc2)
backward_model_pred_err = postResample(pred = backward_model_pred, 
                                       obs = ff_test_proc2$log_area)
summary(backward_model)

#stepwise model
set.seed(1234)
stepwise_model = train(log_area ~ ., 
                       data = ff_train_proc2, 
                       method = "leapSeq",
                       tuneGrid = expand.grid(nvmax = 1:20),
                       #tuneLength = 20,
                       trControl = trainControl(method = "cv", number = 10))
ggplot(stepwise_model) + ggtitle('Stepwise model: nvmax = 1:20')
ggplot(varImp(stepwise_model)) + ggtitle('Feature importantance in stepwise model')

stepwise_model_pred = predict(stepwise_model, newdata = ff_test_proc2)
stepwise_model_pred_err = postResample(pred = stepwise_model_pred, 
                                       obs = ff_test_proc2$log_area)
summary(stepwise_model)

# ridge regression
set.seed(1234)
ridge_model = train(log_area ~ .,
                    data = ff_train_proc2,
                    method = "ridge",
                    #tuneLength = 20,
                    tuneGrid = expand.grid(lambda = seq(0, 1, 0.05)),
                    trControl = trainControl(method = "cv", number = 10))
ggplot(ridge_model) + ggtitle('Ridge model: lambda = 0:1')
ggplot(varImp(ridge_model)) + ggtitle('Feature importance in ridge model')

#plot(ridge_model$finalModel)
# get the model coefficients
ridge_coefs = predict(ridge_model$finalModel, type = "coef")

ridge_model_pred = predict(ridge_model, newdata = ff_test_proc2)
ridge_model_pred_err = postResample(pred = ridge_model_pred, 
                                    obs = ff_test_proc2$log_area)
summary(ridge_model)

# lasso regression
set.seed(1234)
lasso_model = train(log_area ~ ., 
                    data = ff_train_proc2,
                    method = "lasso", 
                    #tuneLength = 20,
                    tuneGrid = expand.grid(fraction = seq(0.05, 1, 0.05)),
                    trControl = trainControl(method = "cv", number = 10))
ggplot(lasso_model) + ggtitle('Lasso model: fraction = 0.05:1')
ggplot(varImp(lasso_model)) + ggtitle('Feature importance in lasso model')

#plot(lasso_model$finalModel)
# get the model coefficients
lasso_coefs = predict(lasso_model$finalModel, type = "coef")

lasso_model_pred = predict(lasso_model, newdata = ff_test_proc2)
lasso_model_pred_err = postResample(pred = lasso_model_pred, 
                                    obs = ff_test_proc2$log_area)
summary(lasso_model)

# model comparison
compare = resamples(list(full_selection = full_model,
                         forward_selection = forward_model,
                         backward_selection = backward_model,
                         stepwise_selection = stepwise_model,
                         ridge_regression = ridge_model,
                         lasso_regression = lasso_model))
summary(compare)
dotplot(compare, main = "Model comparison")

# model performance on test data
df = data.frame(modelType = compare$models, 
                TrainRMSE = 
c(mean(compare$values$`full_selection~RMSE`),
  mean(compare$values$`forward_selection~RMSE`),
  mean(compare$values$`backward_selection~RMSE`),
  mean(compare$values$`stepwise_selection~RMSE`),
  mean(compare$values$`ridge_regression~RMSE`),
  mean(compare$values$`lasso_regression~RMSE`)),
                TestRMSE = 
c(full_model_pred_err[[1]],
  forward_model_pred_err[[1]],
  backward_model_pred_err[[1]],
  stepwise_model_pred_err[[1]],
  ridge_model_pred_err[[1]],
  lasso_model_pred_err[[1]]))
df$diff = df$TrainRMSE-df$TestRMSE
  
# plot for selected final model
errors = data.frame(predicted = forward_model_pred, 
                    observed = ff_test_proc2$log_area, 
                    error = forward_model_pred - ff_test_proc2$log_area)
ggplot(data = errors, aes(x = predicted, y = observed)) +
  #geom_point() +
  geom_jitter(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Observed vs. predicted burn area using forward model') +
  labs(x = "Predicted log_area", y = "Observed log_area") 
