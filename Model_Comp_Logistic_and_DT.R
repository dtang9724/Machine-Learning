# Load packages
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)

# Part I
# Load the dataset
mushroom_raw <- read.csv('https://s3.amazonaws.com/notredame.analytics.data/mushrooms.csv')
mushroom_data <- mushroom_raw

# Data exploration and processing
head(mushroom_data)
glimpse(mushroom_data)
str(mushroom_data$type)
mushroom_data <- mushroom_data %>% 
  mutate(type = ifelse(type=='edible','Yes','No'))
mushroom_data %>% 
  sapply(table) %>% 
  sapply(prop.table)

# Part II
# Create frequency plots for each feature
mushroom_data[,1:12] %>%
  keep(is.factor) %>%
  gather() %>%
  group_by(key,value) %>% 
  summarise(n = n()) %>% 
  ggplot() +
  geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
  coord_flip() +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

mushroom_data[,13:23] %>%
  keep(is.factor) %>%
  gather() %>%
  group_by(key,value) %>% 
  summarise(n = n()) %>% 
  ggplot() +
  geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
  coord_flip() +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

# Looking at the above probability tables and visualizations, we can see that our dependent variable is pretty balanced. 
# This indicates that the less balanced binary features would probably not be good predictors.
# We can eliminate features that have a small number of levels and extremely imbalanced levels.
groups <- as.data.frame(lapply(mushroom_data, nlevels)) %>% 
  pivot_longer(everything(),names_to = 'colnames',values_to = 'lvs')
median(groups$lvs)
mean(groups$lvs)
levels_less_than_median <- groups[groups$lvs <= median(groups$lvs)&groups$lvs <= mean(groups$lvs),]
# The median number of levels is 4 and the mean 5, we should start by eliminating features with less than or equal to 4 levels and imbalance in the number of observations.
# Looking at the plots we created earlier, we can comfortably remove gill_attachment, gill_spacing, gill_size, cap_surface, veil_color, veil_type, and ring_number.
mushroom_data <- mushroom_data %>% 
  select(-gill_attachment, -gill_spacing, -gill_size, -cap_surface, -veil_color, -veil_type, -ring_number)
glimpse(mushroom_data)

# Stratified sampling using caret
set.seed(1234)
sample_set <- createDataPartition(mushroom_data$type, p = .6, list = FALSE)
mushroom_train <- mushroom_data[ sample_set,]
mushroom_test  <- mushroom_data[-sample_set,]

# Part III
# Model selection
# Amongst the available models, we would choose Logistic Regression and Decision Trees.
# Why not choosing KNN: KNN involves calculating Euclidean distance, which requires continuous features. We only have categorical features.
# Why choose Logistic Regression: 
# 1. Logistic Regression does a better job handling categorical variables than KNN. 
# 2. Logistic Regression is useful in predicting a binomial response. This applies to our case.
# Why choose Decision Tree:
# 1. Decision Trees can ignore unimportant features.
# 2. Decision Trees handle categorical variables well.
# 3. The output is easy to understand.
# We first use decision tree because it will help us figure out the most predictive predictor.

# Decision Tree
# Check the proportions for the class between all 3 datasets.
round(prop.table(table(mushroom_data$type)),2)
round(prop.table(table(mushroom_train$type)),2)
round(prop.table(table(mushroom_test$type)),2)
# Our data is pretty balanced. There is no need to use SMOTE.

# Train the model
tree_mod <-
  rpart(
    type ~ .,
    method = "class",
    data = mushroom_train,
    control = rpart.control(cp = 0.001)
  )

# Plot the decision tree using the rpart.plot() function from the rpart.plot library.
rpart.plot(tree_mod)
# We can see that according to the decision tree, the most predictive feature is odor. So we use that for our logistic regression.


# Logistic Regression
# First we look at the levels of each feature.
mushroom_data %>% 
  lapply(levels)

# Notice that population should recoded as ordinal. Also create dummy indicators for type.
mushroom_log <- mushroom_train %>% 
  mutate(population = factor(population, levels=c("solidary", "several", "scattered", "numerous", "clustered", "abundant"), ordered=TRUE)) %>% 
  mutate(type_num = as.integer(ifelse(type == 'Yes',1,0))) %>% 
  select(-type)

# Output of logistic regression between type and odor
logMod <- glm(type_num~odor,family=binomial(link = 'logit'), data = mushroom_log)
summary(logMod)

# Predict based on the test dataset
logit_pred <- predict(logMod, mushroom_test, type = 'response')
head(logit_pred)
logit_pred <- ifelse(logit_pred > 0.5, 1, 0)
head(logit_pred)
logit_pred_table <- table(mushroom_test$type, logit_pred)


# Part IV
# Confusion matrix - logistic regression
logit_pred_table <- table(mushroom_test$type, logit_pred)
logit_pred_table

# Confusion matrix - decision tree
tree_pred <- predict(tree_mod, mushroom_test,  type = "class")
head(tree_pred)
tree_pred_prob <- predict(tree_mod, mushroom_test)
head(tree_pred_prob)
tree_pred_table <- table(mushroom_test$type, tree_pred)
tree_pred_table

# Accuracy Comparison
tree_pred_accuracy <- sum(diag(tree_pred_table)) / nrow(mushroom_test)
tree_pred_accuracy

logit_pred_accuracy <- sum(diag(logit_pred_table)) / nrow(mushroom_test)
logit_pred_accuracy










