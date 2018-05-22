#Data set: winequality.csv
#Your task is to find the best predictors for the wines quality (quality) from the following
#11 variables: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free
#sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol. To that effect, use
#all of the following techniques:
 # - best subset selection regression
#- forward and backward stepwise regression
#- ridge regression
#- lasso regression
#- PLS regression
#Identify the model that provides the best prediction accuracy in the test set.

wine <-read.csv('winequality.csv')

#Check for missisng values 
sapply(wine, function(x) sum(is.na(x)))

library(psych)
pairs.panels(wine)

library(leaps)

#The subset selection is a regression analysis method that 
#consists in choosing, from all the possible predictor subsets,
#the subset that best predicts the dependent variable. 

#The best subset is the subset that gives either the greatest 
#R squared or the smallest mean squared error (MSE).



#Run the Best Subset regression from the leaps package
bss_Fit <- regsubsets(quality~.,wine, nvmax=12)
summ <-summary(bss_Fit)

##To asses the goodness of fit, we use the adjusted R squared
summ$adjr2

## To find the maximum adjusted R squared --11 Is the optimal model
which.max(summ$adjr2)

#Print the coefficients of this model 
coef(bss_Fit, 11)

#The 3 top variables are "alcohol", "sulphates", and volatile.acidity"
#If less variarables and more predictability is desired. 
plot(bss_Fit)

##### Stepwise Forward 
#The forward stepwise selection starts with the null model, 
#then adds predictors to the model one by one. At each stage, 
#the algorithm verifies whether the newly added predictor 
#improves the model. If it does, it is retained in the model; 
#if not, it is dropped.

fsr_Fit <- regsubsets(quality~.,wine, nvmax=12, method="forward")
summ <-summary(bss_Fit)
summ

##To asses the goodness of fit, we use the adjusted R squared
summ$adjr2

## To find the maximum adjusted R squared --11 Is the optimal model
which.max(summ$adjr2)

#Print the coefficients of this model 
coef(fsr_Fit, 11)

plot(fsr_Fit)


###Backward Stepwise Regression 
 
#The backward stepwise selection starts with the full 
#model (i.e. the model that contains all the predictors), 
#then removes the least useful predictors, one by one. 
#At each step, the algorithm evaluates the prediction 
#accuracy of the new model and continues to remove predictors 
#until no improvement in prediction accuracy can be obtained.

bsr_Fit <- regsubsets(quality~.,wine, nvmax=12, method="backward")
summ <-summary(bss_Fit)
summ

##To asses the goodness of fit, we use the adjusted R squared
summ$adjr2

## To find the maximum adjusted R squared --11 Is the optimal model
which.max(summ$adjr2)

#Print the coefficients of this model 
coef(bsr_Fit, 11)

plot(bsr_Fit)

######Ridge Regression
#The penalized regression techniques use all the predictors 
#in the model, but introduce a penalty that constrains 
#(or regularize) the regression coefficients. Some of the 
#coefficients are forced to shrink (decrease towards zero). 

#The penalized techniques are also called shrinkage techniques.

library(glmnet)
## A matix and vector needs to be created

x <- model.matrix(quality~., wine)

y <- wine$quality # Vector of the dependent variables

##Lambda values are power of 10
w <- seq(10,-3, length= 100)
lvalues <-10^w
lvalues

#fit the ridge regression 
rr_fit <- glmnet(x,y, alpha=0, lambda = lvalues)
rr_fit$lambda[40]
coef(rr_fit)[,40]

predict(rr_fit, s=1200, type="coefficients")

plot(rr_fit)

#Validating Ridge reg optimal lambda model with lowest MSE
# Split into training and test set to compute test set mse 

n <-sample(6497, 3240)

#Fit model on the training set 
#Perform 10 fold cross validation 
library(glmnet)

cv_fit <- cv.glmnet(x[n,], y[n], alpha=0, nfolds=10)
## No need to set lambda in the function. 

optLambda <- cv_fit$lambda.min
optLambda

###Predict Y values for the test set
### Using the optimum lambda
### & compute MSE

pred <- predict(cv_fit, s=optLambda, newx=x[-n,])
head(pred)

mse_test <- mean((pred-y[-n])^2)
mse_test

##Lasso Regression 
#The penalized regression techniques use all the predictors in the model, but introduce a penalty that constrains (or regularize) the regression coefficients. Some of the coefficients are forced to shrink (decrease towards zero). 

#The  penalized techniques are also called shrinkage techniques.

#Removes predictors that have little impact

# With higherlambda the greater the penalty

x <- model.matrix(quality~., wine)

y <- wine$quality # Vector of the dependent variables

##Lambda values are power of 10
w <- seq(10,-3, length= 100)
lvalues <-10^w
lvalues

#fit the ridge regression 
lr_fit <- glmnet(x,y, alpha=1, lambda = lvalues)#alpha set to one for lasso
lr_fit$lambda[3]
coef(lr_fit)[,3]

## a model with low lamda 
lr_fit$lambda[99]
coef(lr_fit)[,99]

## Intermediate lambda
lr_fit$lambda[70]
coef(lr_fit)[,70]

## Coefficients different from zero 
coef(lr_fit)[,70][coef(lr_fit)[,70]!=0]

#### Validating lasso regression 

n <-sample(6497, 3240)
## perform k-fold CV

cv_fit <- cv.glmnet(x[n,], y[n], alpha=1, nfolds=10)
optLambda <- cv_fit$lambda.min
optLambda

###Predict Y values for the test set
### Using the optimum lambda
### & compute MSE

pred <- predict(cv_fit, s=optLambda, newx=x[-n,])
head(pred)

mse_test <- mean((pred-y[-n])^2)
mse_test

## Create vector of lambdas and fit lasso again 
w <- seq(10,-3, length=100)
lvalues <- 10^w
lr_fit <- glmnet(x,y, alpha=1, lambda = lvalues)

##print coefficients of the best model 
predict(lr_fit, s=optLambda, type="coefficients")

#### Partial Least Squares Reg
#The PLS regression is a technique that allows us to predict a dependent variable based on a very large set of independent variables. This technique identifies linear combinations of predictors that meet two conditions:
#  they properly explain the initial predictors
#they are related to the response variable as well.
#The optimal number of components - that generates the 
#best model (with the lowest MSE) - is identified through 
#cross-validation.

#The PLS regression is most useful in some fields of 
#research where the number of predictors is greater 
#than the sample size (in cancer research, for example).

library(pls)
#Fit model with k-fold cv
plsr_fit <-plsr(quality~., data =wine, scale=T, validation="CV")
#The variable are standardized 
summary(plsr_fit)
MSE <- 73.47^2
MSE
#5398

#predictor coefficients for the model with 10 components 

coef(plsr_fit, 10)

## Model with optimal numer of components 
plsr_fit2 <- plsr(quality~., data=wine, scale=T, ncomp=10)
summary(plsr_fit2)

#93.53 explained
coef(plsr_fit2)

## Validation PLS regression approach to obtain test MSE
n <-sample(6497, 3240)
wine_train <- wine[n,]
wine_test <-wine[-n,]

##Fit PLS model on the training set 
plsr_fit <- plsr(quality~., data=wine_train, scale=T, ncomp=10)
## Compute the MSE on the test set 

pred <- predict(plsr_fit, wine_test, ncomp=10)
head(pred)

mse <- mean((pred-wine_test$quality)^2)
mse

#MSE=0.543



