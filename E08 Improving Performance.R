#' ---
#' title: "IMPROVING MODEL PERFORMANCE"
#' author: "Fred Nwanganga"
#' date: "April 28th, 2019"
#' ---

# install.packages(c("tidyverse","caret","DMwR","rpart", "ROCR","randomForest","xgboost"))

library(tidyverse)
library(caret)
library(DMwR)
library(rpart)
library(ROCR)
library(randomForest)
library(xgboost)


#----------------------------------------------------------------
#' #1. Collect and Prepare the Data
#----------------------------------------------------------------

loans <- read_csv("https://s3.amazonaws.com/notredame.analytics.data/lendingclub.csv")

loans$Grade <- as.factor(loans$Grade)
loans$EmploymentLength <- as.factor(loans$EmploymentLength)
loans$HomeOwnership <- as.factor(loans$HomeOwnership)
loans$IncomeVerified <- as.factor(loans$IncomeVerified)
loans$LoanPurpose <- as.factor(loans$LoanPurpose)
loans$Default <- as.factor(loans$Default)

# Partition the data using caret's createDataPartition() function.
set.seed(1234)
sample.set <- createDataPartition(loans$Default, p = 0.75, list = FALSE)
loans.train <- loans[sample.set, ]
loans.train <- SMOTE(Default ~ ., data.frame(loans.train), perc.over = 100, perc.under = 200)
loans.test <- loans[-sample.set, ]

#----------------------------------------------------------------
#' #2. Train and Evaluate a Model
#----------------------------------------------------------------
# Going forward, we are going to use the caret package to train and evaluate our models.
# For documentation on caret and its supported methods see 'https://topepo.github.io/caret/'.

# Create a simple model using caret's train() function and the rpart decision tree learner.
tree.mod <- train(Default ~ ., data = loans.train, method = "rpart") # Wrapper makes the rpart function more understandable
# Optimize hyperparameters

# Notice the difference between this approach and what we did in our previous Decision Trees code.

# Look at the results.
tree.mod

# Make predictions based on our candidate model. This still works the same as before.
tree.pred <- predict(tree.mod, loans.test)

# View the Confusion Matrix.
confusionMatrix(tree.pred, loans.test$Default, positive = "Yes")

# Just like before, note that we can obtain predicted classes...
head(predict(tree.mod, loans.test, type = "raw"))

# ...as well as the predicted probabilities (with "raw" and "prob", respectively).
head(predict(tree.mod, loans.test, type = "prob"))

# Note that type = "raw" is the default, so you don't always have to specify it.

#----------------------------------------------------------------
#' #3. Customize the Tuning Process
#----------------------------------------------------------------

# We are now going to use the trainControl() function to create a set of configuration options.
# This is known as a control object.
# We will focus on two things: 
# (1) The resampling strategy (method, number), and 
# (2) The measure used for choosing the best model (selectionFunction). 

ctrl <-
  trainControl(method = "cv", # CROSS-VALIDATION
               number = 10, # 10 FOLD
               selectionFunction = "oneSE") # BEST: BEST PERFORMING; # oneSE: one SE from the best; # tolerance: CONTROL the diff

# There are three options for selectionFunction - {best, oneSE, tolerance}
# Often a "one-standard error" rule is used with cross-validation, in which we choose the most parsimonious model whose error is no more than one standard error above the error of the best model."
# Q: Why not always choose "best"?

# Now let's create a grid of the parameters to optimize.
# Since we are using the rpart learner, we need to get a list of its tuning parameters.
# The modelLookup() function allows us to do this.
modelLookup("rpart")

# Remember that Decision Trees are also implemented using the C5.0 learner.
# If we decided to use that learner, we could also look up its tuning parameters.
modelLookup("C5.0")

# To create the grid of tuning paramaters, we use expand.grid().
# This allows us to fill the grid without having to do it cell by cell.

# For example, to create a search grid for the parameters of C5.0, we do the following.
grid <-
  expand.grid(
    .model = "tree",
    .trials = c(1, 5, 10, 15, 20, 25, 30, 35),
    .winnow = FALSE
  )
class(grid)
# Look at the result of expand.grid().
grid

# Q: What do you think is going to happen here?

# Note the use of the "." notation for the parameter names. This is important.

# Let's create our grid for rpart.
grid <- 
  expand.grid(
    .cp = seq(from=0.0001, to=0.005, by=0.0001) # grid from a to b incremental by
)
grid

# Q: What will happen with this grid?

# Now we can train our model using:
# (1) The control object, 
# (2) Our tuning grid, and 
# (3) Our model performance evaluation statistic (Kappa).

set.seed(1234)
tree.mod <-
  train(
    Default ~ .,
    data = loans.train,
    method = "rpart",
    metric = "Kappa",
    trControl = ctrl,
    tuneGrid = grid
  )

tree.mod

# Note that while the cp = 0.0004 model offers the best raw performance according to Kappa, the cp = 0.0007 model was chosen. 
# It offers similar performance but with a simpler form. Remember Occam's Razor. 
# Simpler methods are not only computationally more efficient, they also reduce the chance of overfitting.


#----------------------------------------------------------------
#' #4. Random Forest
#----------------------------------------------------------------

# For Random Forest, the method we use is called 'rf', so let's look up it's parameters.
modelLookup("rf")

# With the knowledge that .mtry stands for the number of randomly select predictors,
# what is the maximum value for .mtry that we can have for our dataset?

# Create a search grid based on the available parameters.
grid <- expand.grid(.mtry = c(3, 6, 9)) # Random selected features

grid

# Let's create our control object. 
# This time, k=3 for k-fold cross validation and we want the 'best' performing configuration.
ctrl <-
  trainControl(method = "cv",
               number = 3,
               selectionFunction = "best")

# CAUTION: This takes a while to run!
set.seed(1234)
rf.mod <-
  train(
    Default ~ .,
    data = loans.train,
    method = "rf",
    metric = "Kappa",
    trControl = ctrl, # How to resample
    tuneGrid = grid
  )

rf.mod


#----------------------------------------------------------------
#' #5. Extreme Gradient Boosting
#----------------------------------------------------------------

ctrl <-
  trainControl(method = "cv",
               number = 3,
               selectionFunction = "best")

modelLookup("xgbTree")

# There are a lot of parameters to tune here.
# For now, let's simply use the defaults, with some slight variations.

grid <- expand.grid(
  nrounds = 20,
  max_depth = c(4, 6, 8),
  eta =  c(0.1, 0.3, 0.5),
  gamma = 0.01,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.5, 1)
)

grid


# CAUTION: This sometimes takes a while to run!
set.seed(1234)
xgb.mod <-
  train(
    Default ~ .,
    data = loans.train,
    method = "xgbTree",
    metric = "Kappa",
    trControl = ctrl,
    tuneGrid = grid
  )

xgb.mod


#----------------------------------------------------------------
#' #6. Compare Model Performance
#----------------------------------------------------------------
# Using our loans dataset, let's compare the performance of the ensemble methods 
# against those of the logistic and decision tree models we built earlier.

#' ##Logistic Regression
# Train the model.
logit.mod <-
  glm(Default ~ ., family = binomial(link = 'logit'), data = loans.train)

# Use the model to predict outcomes against our test data.
logit.pred.prob <- predict(logit.mod, loans.test, type = 'response')

# Using a decision boundary of 0.5 (i.e If P(y=1|X) > 0.5 then y="Yes" else y="No").
logit.pred <- as.factor(ifelse(logit.pred.prob > 0.5, "Yes", "No"))

test <- loans.test$Default
pred <- logit.pred
prob <- logit.pred.prob

# Plot ROC Curve.
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, main = "ROC Curve for Loan Default Prediction Approaches", col = 2, lwd = 2)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Get performance metrics.
accuracy <- mean(test == pred)
precision <- posPredValue(as.factor(pred), as.factor(test), positive = "Yes")
recall <- sensitivity(as.factor(pred), as.factor(test), positive = "Yes")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- confusionMatrix(pred, test, positive = "Yes")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- tibble(approach="Logistic Regression", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 



#' ##Classification Tree.
tree.pred <- predict(tree.mod, loans.test, type = "raw")
tree.pred.prob <- predict(tree.mod, loans.test, type = "prob")

test <- loans.test$Default
pred <- tree.pred
prob <- tree.pred.prob[,c("Yes")]

# Plot ROC Curve.
# dev.off()
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=3, lwd = 2, add=TRUE)


# Get performance metrics.
accuracy <- mean(test == pred)
precision <- posPredValue(as.factor(pred), as.factor(test), positive = "Yes")
recall <- sensitivity(as.factor(pred), as.factor(test), positive = "Yes")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- confusionMatrix(pred, test, positive = "Yes")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
  add_row(approach="Classification Tree", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 



#' ##Random Forest.
rf.pred <- predict(rf.mod, loans.test, type = "raw")
rf.pred.prob <- predict(rf.mod, loans.test, type = "prob")

test <- loans.test$Default
pred <- rf.pred
prob <- rf.pred.prob[,c("Yes")]

# Plot ROC Curve.
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=4, lwd = 2, add=TRUE)

# Get performance metrics.
accuracy <- mean(test == pred)
precision <- posPredValue(as.factor(pred), as.factor(test), positive = "Yes")
recall <- sensitivity(as.factor(pred), as.factor(test), positive = "Yes")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- confusionMatrix(pred, test, positive = "Yes")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
  add_row(approach="Random Forest", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 


#' ##XGBoost.
xgb.pred <- predict(xgb.mod, loans.test, type = "raw")
xgb.pred.prob <- predict(xgb.mod, loans.test, type = "prob")

test <- loans.test$Default
pred <- xgb.pred
prob <- xgb.pred.prob[,c("Yes")]

# Plot ROC Curve.
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=5, lwd = 2, add=TRUE)

# Get performance metrics.
accuracy <- mean(test == pred)
precision <- posPredValue(as.factor(pred), as.factor(test), positive = "Yes")
recall <- sensitivity(as.factor(pred), as.factor(test), positive = "Yes")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- confusionMatrix(pred, test, positive = "Yes")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
  add_row(approach="Extreme Gradient Boosting", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 


# Draw ROC legend.
legend(0.6, 0.6, c('Logistic Regression', 'Classification Tree', 'Random Forest', 'Extreme Gradient Boosting'), 2:5)


#' ##Output Comparison Table.
comparisons


# Occham's razor: 
# Logistic regression is simpler because RF is an ensemble model, makes more assumptions.
# Logistics regression allows inference - what does each coef mean?
# You can't output the tree for RF