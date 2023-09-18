######### Final Project #########
rm(list=ls())# remove all variables in memory
library(readr)
library(ISLR)
library(boot)
library(olsrr)
install.packages('caTools')
library(caTools)
library(MASS)
library(dplyr)
library(tidyverse)
install.packages("corrplot")
library(corrplot)
install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")
library(glmnet) 
library(reshape)
library(ggplot2)

all_wines <- read_csv("wine-quality-white-and-red.csv")
head(all_wines)

################## Data Exploration ##################
summary(all_wines)
# Most notable: 
# min(quality) = 3 max(quality) = 9 
# residual sugar and free sulfur dioxide has huge outlier 

# Quality Count
all_wines %>% count(quality) 
# Most wines have a quality score of 5 or 6 
# Very few wines have a quality score of 3 or 9 
# Best quality wine has a value of 3.

all_wines <- data.frame(all_wines) %>% 
  mutate(type = ifelse(all_wines$type == "red", yes = 1, no = 0))
pairs(all_wines)
# Doesn't give us information on most significant predictors. 

# Correlation Matrix 
cor_vals <- cor(all_wines)
corrplot(cor_vals)
chart.Correlation(all_wines)

#head(all_wines)
names(all_wines)

################ Model Selection ################
# Forward Selection
model <- lm(quality ~., data=all_wines)
ols_step_forward_p(model)
plot(ols_step_forward_p(model))

# Backward Selection
ols_step_backward_p(model)  
plot(ols_step_backward_p(model) )

# Best Subset Selection
ols_step_best_subset(model)
plot(ols_step_best_subset(model))

############################ Linear Regression ############################
################ k-fold Cross Validation ################
set.seed(17)
# fit a model using all variables
glm.fit = glm(quality~.,data=all_wines)

# customize a cost function to find the abs mean deviation
cost <- function(r, pi) mean(abs(r-pi))
cv.err = cv.glm(all_wines, glm.fit, cost, K=10)
# error rate of all variables
cv.err$delta
#### 0.569924 0.569844

set.seed(17)
# fit a model using best variables
glm.fit2 = glm(quality~ type + fixed.acidity + volatile.acidity + 
                 residual.sugar + chlorides + free.sulfur.dioxide + 
                 total.sulfur.dioxide + density + pH + sulphates + 
                 alcohol, data=all_wines)
cv.err = cv.glm(all_wines, glm.fit2, cost, K=10)
# error rate of best variables
cv.err$delta
#### 0.5699147 0.5698385

############################ Polynomial Regression ############################
names(all_wines)
set.seed(23)
sel <- sample(1:nrow(all_wines), 0.7*nrow(all_wines))

train <- all_wines[sel, ]
valid <- all_wines[-sel, ]
all_wines$typeW = ifelse(all_wines$type == 'white', 1, 0)
all_wines$typeR = ifelse(all_wines$type == 'red', 1, 0)

################## k-fold Cross Validation ################
set.seed(17)
order = 20
bestvars = list("type","fixed.acidity","volatile.acidity",
                "residual.sugar","chlorides","free.sulfur.dioxide", 
                "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")
shuffle = all_wines[sample(nrow(all_wines)),]
K = 10
folds = cut(seq(1,nrow(shuffle)),breaks=K,labels=FALSE)
r.square = matrix(data=NA,nrow=K,ncol=order)
for(i in 1:K){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- shuffle[testIndexes, ]
  trainData <- shuffle[-testIndexes, ]
  for (j in 1:order){
    fit.train = lm(quality ~ I(typeW^j) +I(typeR^j) + I(fixed.acidity^j) +
                     I(volatile.acidity^j) + I(residual.sugar^j) + 
                     I(chlorides^j) + I(free.sulfur.dioxide^j) + 
                     I(total.sulfur.dioxide^j) + I(density^j) + I(pH^j) +
                     I(sulphates^j) + I(alcohol^j), data = trainData)
    fit.test = predict(fit.train, newdata=testData)
    r.square[i,j] = cor(fit.test, testData$quality, use='complete')^2
  }
}

fits.kfold <- colMeans(r.square)
plot(colMeans(r.square), type='l')
print(fits.kfold)
Xmat <- cbind(all_wines$typeW, all_wines$typeR, all_wines$fixed.acidity,
              all_wines$volatile.acidity, all_wines$residual.sugar, 
              all_wines$chlorides, all_wines$free.sulfur.dioxide, 
              all_wines$total.sulfur.dioxide, all_wines$density,
              all_wines$pH, all_wines$sulphates, all_wines$alcohol )
Xmat
colnames(Xmat) <- c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 
                    'x10', 'x11', 'x12')
Xmat_train <- Xmat[sel,]
Xmat_valid <- Xmat[-sel,]
log_lambda_grid <- seq(6, -2, length=100)
lambda_grid <- 10^log_lambda_grid

################## Ridge Regression ################

ridge_models <- glmnet(Xmat_train, train$quality, alpha=0, lambda=lambda_grid)
ridge_coef <- coef(ridge_models)
round(ridge_coef[, c(1:3, 98:100)], 6)

train_pred_ridge <- predict(ridge_models, Xmat_train)
valid_pred_ridge <- predict(ridge_models, Xmat_valid)

valid_pred_ridge[,c(1,2,99,100)]


SSE_train <- colSums((train_pred_ridge - train$quality)^2)
SSE_valid <- colSums((valid_pred_ridge - valid$quality)^2)

SST_train <- var(train$quality) * (nrow(train) - 1)
SST_valid <- var(valid$quality) * (nrow(valid) - 1)

r2_train_list <- 1 - SSE_train / SST_train
r2_valid_list <- 1 - SSE_valid / SST_valid

round(r2_valid_list[c(1:3, 98:100)],4)

plot(log_lambda_grid, r2_train_list, ylim=c(-0.2,1), pch=".", col="salmon",
     xlab="ln(lambda)", ylab="r-Squared", main="Training and Validation Scores (Ridge)")

lines(log_lambda_grid, r2_train_list, col="salmon", lwd=2)
lines(log_lambda_grid, r2_valid_list, col="cornflowerblue", lwd=2)

legend(75, 1, legend=c("Training Acc", "Validation Acc"),
       col=c("salmon", "cornflowerblue"), lty=1, lwd=2, cex=0.8)

best_valid_r2 <- max(r2_valid_list)
best_valid_r2_ix <- which.max(r2_valid_list)
best_log_lambda <- log_lambda_grid[best_valid_r2_ix]

cat('Index of Optimal r-Squared:    ', best_valid_r2_ix, '\n',
    'Value of Optimal r-Squared:    ', best_valid_r2, '\n',
    'Value of Optimal log(lambda):  ', best_log_lambda, sep='')

################## Lasso Regression ##################
Xmat <- cbind(all_wines$typeW, all_wines$typeR, all_wines$fixed.acidity,
              all_wines$volatile.acidity, all_wines$residual.sugar, 
              all_wines$chlorides, all_wines$free.sulfur.dioxide, 
              all_wines$total.sulfur.dioxide, all_wines$density,
              all_wines$pH, all_wines$sulphates, all_wines$alcohol, 
              all_wines$citric.acid)
Xmat
colnames(Xmat) <- c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 
                    'x10', 'x11', 'x12', 'x13')
Xmat_train <- Xmat[sel,]
Xmat_valid <- Xmat[-sel,]
log_lambda_grid <- seq(2, -4, length=100)
lambda_grid <- 10^log_lambda_grid


lasso_models <- glmnet(Xmat_train, train$quality, alpha=1, lambda=lambda_grid)


lasso_coef <- coef(lasso_models)


round(lasso_coef[, c(84, 85)], 1)
train_pred_lasso <- predict(lasso_models, Xmat_train)
valid_pred_lasso <- predict(lasso_models, Xmat_valid)

valid_pred_lasso[,c(1,2,99,100)]

SSE_train <- colSums((train_pred_lasso - train$quality)^2)
SSE_valid <- colSums((valid_pred_lasso - valid$quality)^2)

SST_train <- var(train$quality) * (nrow(train) - 1)
SST_valid <- var(valid$quality) * (nrow(valid) - 1)

r2_train_list <- 1 - SSE_train / SST_train
r2_valid_list <- 1 - SSE_valid / SST_valid

round(r2_valid_list[c(83:87)],4)
plot(log_lambda_grid, r2_train_list, ylim=c(-0.2,1), pch=".", col="salmon",
     xlab="ln(lambda)", ylab="r-Squared", main="Training and Validation Scores (Lasso)")

lines(log_lambda_grid, r2_train_list, col="salmon", lwd=2)
lines(log_lambda_grid, r2_valid_list, col="cornflowerblue", lwd=2)

legend(75, 1, legend=c("Training Acc", "Validation Acc"),
       col=c("salmon", "cornflowerblue"), lty=1, lwd=2, cex=0.8)
best_valid_r2 <- max(r2_valid_list)
best_valid_r2_ix <- which.max(r2_valid_list)
best_log_lambda <- log_lambda_grid[best_valid_r2_ix]

cat('Index of Optimal r-Squared:    ', best_valid_r2_ix, '\n',
    'Value of Optimal r-Squared:    ', best_valid_r2, '\n',
    'Value of Optimal log(lambda):  ', best_log_lambda, sep='')


############################ Ordinal Logistic Regression ############################

## data preparation
# high quality 
all_wines$quality[all_wines$quality<4] <- 1 
# medium quality
all_wines$quality[(all_wines$quality>3) & (all_wines$quality<7) ] <- 2
# low quality
all_wines$quality[all_wines$quality>=7] <- 3
all_wines$quality <- as.factor(all_wines$quality)
summary(all_wines$quality)

set.seed(3000)
spl = sample.split(all_wines$quality, SplitRatio = 0.7)
wine_train = subset(all_wines, spl==TRUE)
wine_test = subset(all_wines, spl==FALSE)

#OLR  with all variables
wine_ord2 <- polr(quality ~ fixed.acidity + volatile.acidity + citric.acid +
                    residual.sugar + chlorides + free.sulfur.dioxide + 
                    total.sulfur.dioxide + density + pH + sulphates + 
                    alcohol, data=wine_train, Hess=TRUE)
summary(wine_ord2)
predict_qual = predict(wine_ord2, all_wines) 
table(all_wines$quality, predict_qual)
mean(as.character(all_wines$quality) != as.character(predict_qual)) 
# Misclassification error
# 0.1897799

# OLR with all but citric.acid since it is not as significant
wine_ord3 <- polr(quality ~ fixed.acidity + volatile.acidity + residual.sugar +
                    chlorides + free.sulfur.dioxide + total.sulfur.dioxide +
                    density + pH + sulphates + alcohol, 
                  data=wine_train, Hess=TRUE)
summary(wine_ord3)
predict_qual1 = predict(wine_ord3, all_wines)
table(all_wines$quality, predict_qual1)

mean(as.character(all_wines$quality) != as.character(predict_qual1))
# Misclassification error
# 0.1891642








