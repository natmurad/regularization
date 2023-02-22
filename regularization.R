#####   Name: regularization.R
#####   Version: 1.0
#####   Description: Regularization methods for feature selection
#####   Date of Creation: Fev-2023
#####   Author: Natália Faraj Murad
#####   E-MAIL: nataliafmurad@gmail.com
#####   PROJECT: https://github.com/natmurad/reg

############################
###--- Load libraries ---###
############################
library(glmnet)
library(ggplot2)
library(caret)
library(impute) # impute missing values
#library(lars)
library(reshape2)
library(forecast)
library(UBL)
library(gridExtra)
library(pls)

############################
###--- Set Directory  ---###
############################
setwd("~/Folder/data")

############################
###--- Read the Data  ---###
############################
data <- read.csv("data.csv", h = F)
metnames <- data[1,]

data1 <- read.csv("meatdata.csv")

#############################
###--- Format the Data ---###
#############################
colnames(data) <- data[1,]
rownames(data) <- data[,1]

data <- data[2:nrow(data), 2:ncol(data)]

data_num <- apply(data, 2, as.numeric)
rownames(data_num) <- rownames(data)
colnames(data_num) <- colnames(data)

data <- data_num
rm(data_num)

############################
###---    Filtering   ---###
############################

# Filter metabolites with more than 40% missing values
NA_percent <- apply(data, 1, function(vec){sum(is.na(vec))/ncol(data)})
data <- data[which(NA_percent<0.4),]

# Select overlap
rownames <- rowData[,'features'] # rows are the features
#metnames <- t(metnames)

#common <- metnames[metnames %in% rownames]


#data <- data[, c(which(colnames(data) %in% common))]



#rownames(data) <- data[1,]

############################
###--- Imputation NA  ---###
############################
data_imp <- impute.knn(data)
data_imp <- as.data.frame(data_imp$data)

data_imp <- log(data_imp)

############################
###--- Scale the Data ---###
############################
paretoscale <- function(z){
  rowmean <- apply(z,1,mean)
  rowsd <- apply(z,1,sd)
  rowsqrtsd <- sqrt(rowsd)
  rv <- sweep(z,1,rowmean,"-")
  rv <- sweep(rv,1,rowsqrtsd,"/")
  return(rv)
}

data_imp <- paretoscale(data_imp)

#data_qn <- preprocessCore::normalize.quantiles(as.matrix(data_imp))
#colnames(data_qn) = colnames(data)
#rownames(data_qn) = rownames(data)

# response variable - to be predicted
age <- data1 %>%
  dplyr::select(public_client_id, age)

rownames(age) <- age[,1]
age$public_client_id <- NULL


data_merged <- merge(data_imp, age, by = 0) # merge response var + imputed features

rownames(data_merged) <- data_merged$Row.names
data_merged$Row.names <- NULL

################################
###--- Split train & test ---###
################################
set.seed(150)
index = sample(1:nrow(data_merged), 0.8*nrow(data_merged)) 

x_tr = data_merged[index,] # Create the training data 
x_ts = data_merged[-index,]

##############################
###--- Balance the data ---###
##############################
# not always needed - Optional

x_tr_balanced <- SMOGNRegress(age~., x_tr, rel = "auto", thr.rel = 0.5,
                              C.perc = list(1.8,0.8), k = 5, repl = TRUE,
                              dist = "p-norm", p = 2, pert=0.1)



hist(x_tr$age)
shapiro.test(log(x_tr$age))

hist(x_tr_balanced$age)
shapiro.test(x_tr_balanced$age)
table(x_tr_balanced$age)
table(x_tr$age)


#x_tr <- x_tr_balanced

###############################
###--- Linear Regression ---###
###############################

linear_reg <- lm(formula = age ~ ., data = x_tr)
summary(linear_reg)

####################################
###--- Lasso/Ridge Regression ---###
####################################

## α=1 is lasso regression (default) and α=0 is ridge regression
## choose type.measure better for your problem
## type.measure = c("default", "mse", "deviance", "class", "auc", "mae", "C")
## also try different nfolds values

# Test to find best parameters -  kfold cross validation
cv_model <- cv.glmnet(as.matrix(x_tr[,-ncol(x_tr)]),
                      as.matrix(x_tr[,'age']), alpha = 1, nlambda = 1000,
                      nfolds=25, standardize = TRUE, type.measure = "mse")



best_lambda <- cv_model$lambda.min
best_lambda
#R_Squared =  max(1 - cv_model$cvm/var(x_tr[,'age']))
plot(cv_model)
print(cv_model)

# Once you found best parameters, adjusts the model using it
best_model <- glmnet(as.matrix(x_tr[,-ncol(x_tr)]),
                     as.matrix(x_tr[,'age']),
                     alpha = 1, lambda = best_lambda, standardize = TRUE)

plot(best_model, xvar = "norm", label = TRUE)

predict(best_model, s = best_lambda, newx = as.matrix(x_ts[,-ncol(x_ts)]))

best_model$dev.ratio
coef(best_model)


## evaluation
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  MAE = sum(abs(predicted - true))/length(predicted)
  
  
# Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square,
    MAE = MAE
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(best_model, s = best_lambda, newx = as.matrix(x_tr[,-(ncol(x_tr))]))
eval_results(x_tr[,ncol(x_tr)], predictions_train, x_tr)

# Prediction and evaluation on test data
predictions_test <- predict(best_model, s = best_lambda, newx = as.matrix(x_ts[,-(ncol(x_ts))]))
eval_results(as.matrix(x_ts[,(ncol(x_ts))]), predictions_test, x_ts)


#find SST and SSE
#sst <- sum((as.matrix(x_ts[,(ncol(x_ts))]) - mean(as.matrix(x_ts[,(ncol(x_ts))])))^2)
#sse <- sum((predictions_test - as.matrix(x_ts[,(ncol(x_ts))]))^2)

#find R-Squared
#rsq <- 1 - sse/sst
#rsq


####################################
###---      ElasticNet        ---###
####################################

# Set training control
#train_cont <- trainControl(method = "repeatedcv",
#                           number = 10,
#                           repeats = 5,
#                         search = "random",
#                          verboseIter = TRUE)

# Train the model
#elastic_reg <- train(age ~ .,
#                    data = x_tr,
#                    method = "glmnet",
#                    preProcess = c("pca"),
#                    tuneLength = 10,
#                    trControl = train_cont)


# Best tuning parameter
#elastic_reg$bestTune

# Make predictions on training set
#predictions_train <- predict(elastic_reg, x_tr[,-ncol(x_tr)])
#eval_results(as.matrix(x_tr[,ncol(x_tr)]), predictions_train, x_tr) 

# Make predictions on test set
#predictions_test <- predict(elastic_reg, x_ts[,-ncol(x_ts)])
#eval_results(as.matrix(x_ts[,ncol(x_ts)]), predictions_test, x_ts)



# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
#cat('Min Lambda: ', cv_model$lambda.min, '\n 1Sd Lambda: ', cv_model$lambda.1se)

#########################################
###---      Select Features        ---###
#########################################
df_coef <- round(as.matrix(coef(best_model, s=cv_model$lambda.min)), 2)

# See all contributing variables
#selected <- df_coef[df_coef[, 1] != 0, ]
selected <- df_coef[abs(df_coef[, 1]) > 0.2, ] ### choose your cutoff

data_merged_sel <- data_merged[,c(which(colnames(data_merged) %in% names(selected)))]


data_merged_sel <- cbind(data_merged_sel, age = data_merged$age)
