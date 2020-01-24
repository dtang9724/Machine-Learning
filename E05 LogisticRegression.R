#' ---
#' title: "LOGISTIC REGRESSION - Identifying Risky Bank Loans"
#' author: "Fred Nwanganga"
#' date: "June 3rd, 2019"
#' ---

# install.packages(c("tidyverse","gridExtra","corrplot","DMwR","InformationValue"))

library(tidyverse)


#----------------------------------------------------------------
#' #1. Collect the Data
#----------------------------------------------------------------

loans <- read_csv("https://s3.amazonaws.com/notredame.analytics.data/lendingclub.csv")

glimpse(loans)

#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------

#' Convert Grade, EmploymentLength, HomeOwnership, IncomeVerified, LoanPurpose and Default to factors.
loans$Grade <- as.factor(loans$Grade)
loans$EmploymentLength <- as.factor(loans$EmploymentLength)
loans$HomeOwnership <- as.factor(loans$HomeOwnership)
loans$IncomeVerified <- as.factor(loans$IncomeVerified)
loans$LoanPurpose <- as.factor(loans$LoanPurpose)
loans$Default <- as.factor(loans$Default)

# Recode the class levels to 0/1.
loans <- loans %>%
  mutate(Default = recode(Default, "No" = "0")) %>%
  mutate(Default = recode(Default, "Yes" = "1"))

summary(loans)


#----------------------------------------------------------------
#' ##Numeric Features
#----------------------------------------------------------------

#' Let's take a look at the data distributions for the numeric features.
loans %>%
  keep(is.numeric) %>% # ONLY SELECT NUMERIC FEATURES
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()


#----------------------------------------------------------------
#' ###Dealing with Outliers
#----------------------------------------------------------------

# Q: Which features seem to have issues with outlier data?

# Let's begin by looking at AnnualIncome.
# We can represent the data distribution as a scatter plot
p1 <- ggplot(data=loans) +
  geom_point(mapping=aes(x=seq(AnnualIncome), y=AnnualIncome))

#' ... and as a box plot.
p2 <- ggplot(data=loans) +
  geom_boxplot(mapping=aes(x="Annual Income", y=AnnualIncome))

# Using the grid.arrange() function from the gridExtra package 
# allows us to plot multiple charts in the same window.
library(gridExtra)
grid.arrange(p1, p2, ncol=2)
 

# We definitely have a problem with outliers in our data.
# There are two common ways to deal with outliers:
# (1) We can either use expert judgement or (2) a rule of thumb.
# The simple rule of thumb approach states that any values greater than 
# Q3 + 1.5 * IQR is an outlier and should be removed.

# Let's use the rule of thumb AnnualIncome.
loans <- loans %>%
  filter(AnnualIncome <= quantile(AnnualIncome, .75) + (1.5 * IQR(AnnualIncome)))

# Recreate the plots to see what happened to the data distribution.
p3 <- ggplot(data=loans) +
  geom_point(mapping=aes(x=seq(AnnualIncome), y=AnnualIncome))

p4 <- ggplot(data=loans) +
  geom_boxplot(mapping=aes(x="Annual Income", y=AnnualIncome))

grid.arrange(p1, p2, p3, p4, ncol=2)


# Now, let's remove the outliers for RevolvingCredit as well.
loans <- loans %>%
  filter(RevolvingCredit <= quantile(RevolvingCredit, .75) + (1.5 * IQR(RevolvingCredit)))


#----------------------------------------------------------------
#' ###Dealing with Low Information Value
#----------------------------------------------------------------

#' Let's get another look at our data distributions.
loans %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

# Q: Which features seem to have issues with low information value?

# Take a look at the data distribution for Delinquencies.
round(prop.table(table(select(loans,Delinquencies))),4) * 100

#' Almost 90% of the Delinquencies are 0. This feature doesn't have much information value, so we need to get rid of it.
loans <- select(loans, -Delinquencies)

# Let's take a look at Inquiries, LoanTerm and PublicRecords.
round(prop.table(table(select(loans,Inquiries))),4) * 100
round(prop.table(table(select(loans,LoanTerm))),4) * 100
round(prop.table(table(select(loans,PublicRecords))),4) * 100

# Let's get rid of PublicRecords.
loans <- select(loans, -PublicRecords)

#' What do our distributions now look like?
loans %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

#----------------------------------------------------------------
#' ###Dealing with Multicollinearity #Drop one of collineared OR Use VIF??
#----------------------------------------------------------------

#' How do each of the numeric fatures correlate with each other?
#' We use the cor() and corrplot() functions from the corrplot package for this.
library(corrplot)
loans %>%
  keep(is.numeric) %>%
  cor() %>%
  corrplot()

#' The chart shows that LoanAmount and Installment are highly correlated.
#' We don't need to use both, so we get rid of Installment.
loans <- select(loans, -Installment)

#' The same applies to TotalAccounts and OpenAccounts.
#' We get rid of TotalAccounts.
loans <- select(loans, -TotalAccounts)


#----------------------------------------------------------------
#' ##Categorical Features
#----------------------------------------------------------------

# Now let's visualize our categorical features. # IMBALANCED PROBLEM
loans %>%
  keep(is.factor) %>%
  gather() %>%
  group_by(key,value) %>% 
  summarise(n = n()) %>% 
  ggplot() +
  geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
  coord_flip() +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

# Nothing much to see here. Moving on...
# Wait! Not so fast. I think we have a data quality issue.

# Q: What data quality issue do we see here?

#----------------------------------------------------------------
#' ##Create training and test datasets
#----------------------------------------------------------------

# Using a 75% to 25% split, we create our training and test datasets.
set.seed(1234)
sample_set <- sample(nrow(loans), round(nrow(loans)*.75), replace = FALSE)
loans_train <- loans[sample_set, ]
loans_test <- loans[-sample_set, ]

# What is the class distribution for each of our datasets?
round(prop.table(table(select(loans, Default), exclude = NULL)), 4) * 100
round(prop.table(table(select(loans_train, Default), exclude = NULL)), 4) * 100
round(prop.table(table(select(loans_test, Default), exclude = NULL)), 4) * 100


# To make our model more generalizable, we need to balance our training data.
# To do this, we use the SMOTE() function from the DMwR package. 
# Synthetic Minority Oversampling Technique: BALANCE THE TRAINING (ONLY) DATASET (A BIAS)
# USE ALL INDEPENDENT VARIABLES TO PREDICT THE RESPONSE = DEFAULT
# Oversample by 100%, undersample by 200%
# FOR TRAINING, CREATE 100% MORE ON CLASS IMBLANCED, LESSEN SAMPLES FROM MORE BALANCED BY 200% OF EXISTING TRAINING
library(DMwR)
loans_train <- SMOTE(Default ~ ., data.frame(loans_train), perc.over = 100, perc.under = 200)

# What is the class distribution for each of our datasets now?
round(prop.table(table(select(loans, Default), exclude = NULL)), 4) * 100
round(prop.table(table(select(loans_train, Default), exclude = NULL)), 4) * 100
round(prop.table(table(select(loans_test, Default), exclude = NULL)), 4) * 100

# Q: Why do we only balance our training data?

# We are now ready to build our model.


#----------------------------------------------------------------
#' #3. Train the Model
#----------------------------------------------------------------

#' Using the glm() function from the stats package, we can build a logistic regression model.
logit_mod <-
  glm(Default ~ ., family = binomial(link = 'logit'), data = loans_train)

#' View the results of the model.
summary(logit_mod)

# Q: Are all the features in our model significant?

# Q: How do we interpret the coefficients?

# View the model coefficients in exponent form.
exp(coef(logit_mod)["LoanAmount"])
# AS LOANAMOUNT INCREASES BY, THE ODSS OF DEFAULT INCREASES BY 1.000003
# IF LESS THAN 1, THE ODDS OF LAMT GO DOWN
#----------------------------------------------------------------
#' #4. Evaluate the Model's Performance
#----------------------------------------------------------------

#' Generate predictions against the data using the model.
logit_pred <- predict(logit_mod, loans_test, type = 'response')
head(logit_pred)
# USE MODEL TO PREDICT TEST DATA. 

# It is common to use a decision boundary of 0.5 to interpret the predictions.
# However, this cutoff value is not always appropriate in all cases.
# We will make use of a function called optimalCutoff() from the InformationValue package
# to get the best decison boundary for our specific model.
library(InformationValue)
ideal_cutoff <-
  optimalCutoff(
    actuals = loans_test$Default,
    predictedScores = logit_pred,
    optimiseFor = "Both"
  )

# What is the decision boundary?
ideal_cutoff

#' Use the decision boundary to transform the results (if P(y=1|X) > ideal_cutoff then 1 else 0).
logit_pred <- ifelse(logit_pred > ideal_cutoff, 1, 0)
head(logit_pred)

# Using our predictions, we can construct a Confusion Matrix.
logit_pred_table <- table(loans_test$Default, logit_pred)
logit_pred_table

# What is our accuracy?
sum(diag(logit_pred_table)) / nrow(loans_test)


#----------------------------------------------------------------
#' #5. Improve the Model's Performance
#----------------------------------------------------------------

#' Let's revisit the summary output of our model.
summary(logit_mod)

#' Based on our previous results, we'll create a new model with only the significant features.
logit_mod2 <-
  glm(
    Default ~ . -LoanAmount -DTI -OpenAccounts -RevolvingCredit,
    family = binomial(link = 'logit'),
    data = loans_train
  )

summary(logit_mod2)

# Make predictions.

logit_pred2 <- predict(logit_mod2, loans_test, type = 'response')
loans_test = loans_test %>% 
  mutate(logit_pred2=logit_pred2)
loans_test %>% 
  ggplot(., aes(InterestRate, logit_pred2)) +
  geom_line(size = 1.5) +
  theme_minimal()
# Get ideal cutoff.
ideal_cutoff2 <-
  optimalCutoff(
    actuals = loans_test$Default,
    predictedScores = logit_pred2,
    optimiseFor = "both"
  )

# What is the decision boundary?
ideal_cutoff2

# Transformm predictions based on ideal cutoff.
logit_pred2 <- ifelse(logit_pred2 > ideal_cutoff2, 1, 0)

#' What does our confusion matrix now look like?
logit_pred2_table <- table(loans_test$Default, logit_pred2)
logit_pred2_table

#' Now, what is our accuracy?
sum(diag(logit_pred2_table)) / nrow(loans_test)


#' So, which model is better?

