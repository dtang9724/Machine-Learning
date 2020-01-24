#' ---
#' title: "DECISION TREES - Identifying Risky Bank Loans"
#' author: "Fred Nwanganga"
#' date: "April 14th, 2019"
#' ---

# install.packages(c("tidyverse","rpart","rpart.plot","DMwR"))

library(tidyverse)
library(rpart)
library(rpart.plot)
library(DMwR)

#----------------------------------------------------------------
#' #1. Collect the Data
#----------------------------------------------------------------

loans <- read_csv("https://s3.amazonaws.com/notredame.analytics.data/lendingclub.csv")

# Convert Grade, EmploymentLength, HomeOwnership, IncomeVerified, LoanPurpose and Default to factors.
loans$Grade <- as.factor(loans$Grade)
loans$EmploymentLength <- as.factor(loans$EmploymentLength)
loans$HomeOwnership <- as.factor(loans$HomeOwnership)
loans$IncomeVerified <- as.factor(loans$IncomeVerified)
loans$LoanPurpose <- as.factor(loans$LoanPurpose)
loans$Default <- as.factor(loans$Default)

#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------

summary(loans)

# Using the sample() function, let's create our training and test datasets.
set.seed(1234)
sample_set <- sample(nrow(loans), round(nrow(loans)*.75), replace = FALSE)
loans_train <- loans[sample_set, ]
loans_test <- loans[-sample_set, ]

# From our summary statistics, we know that our data set is imbalanced.
# To make our model more generalizable, we need to learn with a more balanced data set.
# To do this, we use the SMOTE() function from the DMwR package.

# Generate new balanced training data.
set.seed(1234)
loans_train <- SMOTE(Default ~ ., data.frame(loans_train), perc.over = 100, perc.under = 200)

# Check the proportions for the class between all 3 datasets.
round(prop.table(table(loans$Default)),2)
round(prop.table(table(loans_train$Default)),2)
round(prop.table(table(loans_test$Default)),2)

# NOTE: Class imbalance is a particularly problematic issue with decision trees.

#----------------------------------------------------------------
#' #3. Train the Model
#----------------------------------------------------------------

tree_mod <-
  rpart(
    Default ~ .,
    method = "class",
    data = loans_train,
    control = rpart.control(cp = 0.005)
  )

# Plot the decision tree using the rpart.plot() function from the rpart.plot library.
rpart.plot(tree_mod)
# The decimal in the second row means how many yes do we have in this node

#----------------------------------------------------------------
#' #4. Evaluate the Model's Performance
#----------------------------------------------------------------

# Make predictions using our tree model against the test set.
tree_pred <- predict(tree_mod, loans_test,  type = "class")

head(tree_pred)

# Note that if we exclude type="class", we get predicted probabilities.
tree_pred_prob <- predict(tree_mod, loans_test)
head(tree_pred_prob)

# Using our predictions, we can construct a Confusion Matrix.
tree_pred_table <- table(loans_test$Default, tree_pred)
tree_pred_table

# What is our accuracy?
tree_pred_accuracy <- sum(diag(tree_pred_table)) / nrow(loans_test)
tree_pred_accuracy


#----------------------------------------------------------------
#' #5. Improve the Model's Performance
#----------------------------------------------------------------

# LOSS/COST MATRIX
# We can include a loss matrix to our model in order to change the relative importance 
# of misclassifying a default as non-default versus a non-default as a default. 
# The goal here is to stress that misclassifying a default as a non-default should be penalized more heavily.

# Create a loss matrix.
matrix(c(0, 2, 1, 0), ncol = 2)

# What do the cells mean?

# Apply the loss matrix to our model.
tree_mod_loss <-
  rpart(
    Default ~ .,
    method = "class",
    data = loans_train,
    parms = list(loss = matrix(c(0, 2, 1, 0), ncol = 2)), ## Parms argument to include loss
    control = rpart.control(cp = 0.005)
  )

# Make predictions using our new tree model against the test set.
tree_pred_loss <- predict(tree_mod_loss, loans_test,  type = "class")

# Show Confusion Matrix...
tree_pred_loss_table <- table(loans_test$Default, tree_pred_loss)
tree_pred_loss_table

# ...and accuracy.
tree_pred_loss_accuracy <- sum(diag(tree_pred_loss_table)) / nrow(loans_test)
tree_pred_loss_accuracy

