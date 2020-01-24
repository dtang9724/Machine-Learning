#' ---
#' title: "EVALUATING MODEL PERFORMANCE"
#' author: "Fred Nwanganga"
#' date: "June 5th, 2019"
#' ---

# install.packages(c("tidyverse","rpart","DMwR","caret","e1071","ROCR"))

library(tidyverse)
library(rpart)
library(DMwR)
library(caret) ## 
library(e1071) ## Helper, naive bayes
library(ROCR) ## ROC Curve


#----------------------------------------------------------------
#' #1. Collect the Data, Prepare the Data and Train the Model
#----------------------------------------------------------------

loans <- read_csv("https://s3.amazonaws.com/notredame.analytics.data/lendingclub.csv")

loans$Grade <- as.factor(loans$Grade)
loans$EmploymentLength <- as.factor(loans$EmploymentLength)
loans$HomeOwnership <- as.factor(loans$HomeOwnership)
loans$IncomeVerified <- as.factor(loans$IncomeVerified)
loans$LoanPurpose <- as.factor(loans$LoanPurpose)
loans$Default <- as.factor(loans$Default)

# Remove unneeded features.
loans$Delinquencies <- NULL
loans$PublicRecords <- NULL
loans$Installment <- NULL
loans$TotalAccounts <- NULL

# Partition the data and balance the training data.
set.seed(1234)
sample.set <- sample(nrow(loans), round(nrow(loans)*.75), replace = FALSE)
loans.train <- loans[sample.set, ]
loans.train <- SMOTE(Default ~ ., data.frame(loans.train), perc.over = 100, perc.under = 200)
loans.test <- loans[-sample.set, ]

# Train the model.
tree.mod <-
  rpart(
    Default ~ .,
    method = "class",
    data = loans.train,
    control = rpart.control(cp = 0.005)
  )

# Use the model to predict outcomes against our test data.
tree.pred <- predict(tree.mod, loans.test,  type = "class")
head(tree.pred)

tree.pred.prob <- predict(tree.mod, loans.test,  type = "prob") 
head(tree.pred.prob) # Prepensity????

#----------------------------------------------------------------
#' #2. Performance Evaluation Metrics
#----------------------------------------------------------------

# We've previously used the table() function to create a Confusion Matrix.
# We can also do the same using the confusionMatrix() function from the caret package.
# The output from this function gives some additional measures to consider.
tree.matrix <- confusionMatrix(tree.pred, loans.test$Default, positive = "Yes") 
tree.matrix

# We can get the Accuracy and Kappa stats from the confusion matrix by using the 'overall' attribute.
tree.accuracy <- as.numeric(tree.matrix$overall["Accuracy"])
tree.accuracy
tree.kappa <- as.numeric(tree.matrix$overall["Kappa"])
tree.kappa

# We can get Sensitivity, Specificity, Precision and Recall directly using different functions.
tree.sensitivity <- sensitivity(tree.pred, loans.test$Default, positive = "Yes")
tree.sensitivity
tree.specificity <- specificity(tree.pred, loans.test$Default, negative = "No")
tree.specificity
tree.precision <- posPredValue(tree.pred, loans.test$Default, positive = "Yes")
tree.precision
tree.recall <- tree.sensitivity
tree.fmeasure <- (2 * tree.precision * tree.recall)/(tree.precision + tree.recall)
tree.fmeasure

# Note the use of the 'positive' and 'negative' parameters above.

#----------------------------------------------------------------
#' #3. Visualization of Performance
#----------------------------------------------------------------

# Visualizations, such as the ROC curve, are useful in performance evaluation.
# To generate an ROC curve from our predictions, we use the ROCR package.

# First, we create a prediction object...
roc.pred <- prediction(predictions = tree.pred.prob[,"Yes"], labels = loans.test$Default) # from rocker

# ...then a performance object.
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")

# Now, we can plot the ROC curve.
plot(roc.perf, main = "ROC Curve for Loan Default Predictions", col = 2, lwd = 3)

# With a reference line.
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Using the performance object, we can also get the Area Under the Curve (AUC)
auc <- performance(roc.pred, measure = "auc") %>%
  slot("y.values") %>% # Bring stuff out of the slot object
  unlist() # Change vector to numeric value
auc



#----------------------------------------------------------------
#' #4. Data Partitioning - Cross-Validation
#----------------------------------------------------------------

# The createFolds() function in the caret package is useful for doing k-fold cross-validation.
# Note that this uses the stratified cross-validation approach.

# Reset the training data (imbalanced version).
loans.train <- loans[sample.set, ]

# Create folds.
set.seed(1234)
folds <- createFolds(loans.train$Default, k = 5)

# See list of vectors containing row numbers for k=5.
glimpse(folds)

# Create training and test(validation) sets using fold 1.
loans.kfold.train <- loans.train[-folds$Fold1,]
loans.kfold.test <- loans.train[folds$Fold1,]

# With k=5, we would need to perform this process 5 times.
# This is a tedious process that could benefit from some automation.

# Run through and generate Kappa stats for each fold. This takes a bit.
kfold.results <- map(folds, function(x) {
  loans.kfold.train <- loans.train[-x, ]
  loans.kfold.train <- SMOTE(Default ~ ., data.frame(loans.kfold.train), perc.over = 100, perc.under = 200)
  loans.kfold.test <- loans.train[x, ]
  tree.kfold.mod <-
    rpart(
      Default ~ .,
      method = "class",
      data = loans.kfold.train,
      control = rpart.control(cp = 0.005)
    )
  tree.kfold.pred <- predict(tree.kfold.mod, loans.kfold.test,  type = "class")
  tree.kfold.matrix <- confusionMatrix(tree.kfold.pred, loans.kfold.test$Default, positive = "Yes")
  tree.kfold.kappa <- as.numeric(tree.kfold.matrix$overall["Kappa"])
  return(tree.kfold.kappa)
})

# View the statistics.
glimpse(kfold.results)

# What is our mean performance?
mean(unlist(kfold.results))


#----------------------------------------------------------------
#' #5. Evaluate the Performance of Several Learners
#----------------------------------------------------------------

# Balance the training data.
loans.train <- loans[sample.set, ]
loans.train <- SMOTE(Default ~ ., data.frame(loans.train), perc.over = 100, perc.under = 200)

# Clear plot window.
dev.off()

# DECISION TREES
test <- loans.test$Default
pred <- tree.pred
prob <- tree.pred.prob[,c("Yes")]

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
#kappa <- kappa2(data.frame(test, pred))$value
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- tibble(approach="Classification Tree", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 

confusionMatrix(tree.pred, loans.test$Default, positive = "Yes") 

# LOGISTIC REGRESSION
# Train the model.
logit.mod <-
  glm(Default ~ ., family = binomial(link = 'logit'), data = loans.train)

# Use the model to predict outcomes against our test data.
logit.pred.prob <- predict(logit.mod, loans.test, type = 'response')

# Using a decision boundary of 0.5 (i.e If P(y=1|X) > 0.5 then y="Yes" else y="No").
logit.pred <- ifelse(logit.pred.prob > 0.5, "Yes", "No")

test <- loans.test$Default
pred <- logit.pred
prob <- logit.pred.prob

# Plot ROC Curve.
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
#kappa <- kappa2(data.frame(test, pred))$value
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
  add_row(approach="Logistic Regression", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 

# Draw ROC legend.
legend(0.6, 0.6, c('Classification Tree', 'Logistic Regression'), 2:3)

# Output comparison table.
comparisons

# Though CT's accuracy is better, we choose LR because of larger AUC.

