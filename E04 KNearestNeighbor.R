#' ---
#' title: "K NEAREST NEIGHBOR - Wisconsin Breast Cancer Biopsy Dataset"
#' author: "Fred Nwanganga"
#' date: "June 3rd, 2019"
#' ---

# install.packages(c("tidyverse", "class"))
library(tidyverse)
library(class)



#----------------------------------------------------------------
#' #1. Collect the Data
#----------------------------------------------------------------

# Let's start by importing the dataset.
wbcd <- read_csv("https://s3.amazonaws.com/notredame.analytics.data/wisconsinbiopsydata.csv")

# Preview the data.
glimpse(wbcd)
head(wbcd)

#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------
# We do not need the 'id' feature so we can drop it.
# We should also convert the 'diagnosis' feature to a factor (a required step for several R ML classifiers).
wbcd <- wbcd %>%
  select(-id) %>%
  mutate(diagnosis = factor(diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant")))

         
# Now let's take a look at the distribution of the diagnoses using the table() function.
table(select(wbcd,diagnosis))


# As we've seen before, we can also show the distribution as proportions.
round(prop.table(table(select(wbcd,diagnosis))) * 100, digits = 1)


# Let's take a look at the descriptive stats for some of the features.
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])


# Notice the significant difference in scales.


# We need to normalize our scales in order not to skew our distance measures.
# Notice that we do not include the class feature ("diagnosis") in the process.
wbcd_z <- as.data.frame(scale(wbcd[-1]))



# Confirm that the transformation was applied correctly.
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
summary(wbcd_z[c("radius_mean", "area_mean", "smoothness_mean")])


# Tibbles are useful in data wrangling because of some of the safeguards built into them.
# However, a number of machine learning algorithms struggle with tibbles.
# To avoid this, we need to convert our data to a data frame prior to the modeling process.
wbcd <- data.frame(wbcd)


# Now we can split our data set into training and test subsets.
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ]


# We also do the same with the class labels.
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]


#----------------------------------------------------------------
#' #3. Train the Model
#----------------------------------------------------------------

# Let's classify the test data with k set to 21 (approx. the square root of 469).
# To do this, we use the knn() function from the class package.
wbcd_test_pred <-
  knn(
    train = wbcd_train,
    test = wbcd_test,
    cl = wbcd_train_labels,
    k = 21
  )


# What did we predict?
wbcd_test_pred


#----------------------------------------------------------------
#' #4. Evaluate the Model's Performance
#----------------------------------------------------------------

# Using our predictions, we can construct a Confusion Matrix.
wbcd_pred_table <- table(wbcd_test_labels, wbcd_test_pred)
wbcd_pred_table

# What is our accuracy?
sum(diag(wbcd_pred_table)) / nrow(wbcd_test)


#----------------------------------------------------------------
#' #5. Improve the Model's Performance
#----------------------------------------------------------------

# To try to improve our model's performance, let's try using 
# min-max normalization instead of z-score standardization.
# This should minimize further the impact of outliers in our distance calculation.
# The first thing we do is create the min-max normalization function.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Then we use the map() function from the purr package (part of the Tidyverse) to apply 
# the normalization to all the features of the data, except the class feature ("diagnosis").
wbcd_n <- as.data.frame(map(wbcd[2:31], normalize))



# Compare the pre- and post-normalization stats.
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
summary(wbcd_n[c("radius_mean", "area_mean", "smoothness_mean")])


# Re-create the training and test datasets.
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]


# Re-classify with the new data.
wbcd_test_pred <-
  knn(
    train = wbcd_train,
    test = wbcd_test,
    cl = wbcd_train_labels,
    k = 21
  )


# What do our results now look like?
wbcd_pred_table <- table(wbcd_test_labels, wbcd_test_pred)
wbcd_pred_table
sum(diag(wbcd_pred_table)) / nrow(wbcd_test)


# Let's try to improve our accuracy by varying the values of k.

# k=1
wbcd_test_pred <-
  knn(
    train = wbcd_train,
    test = wbcd_test,
    cl = wbcd_train_labels,
    k = 1
  )

wbcd_pred_table <- table(wbcd_test_labels, wbcd_test_pred)
wbcd_pred_table
sum(diag(wbcd_pred_table)) / nrow(wbcd_test)

# k=5
wbcd_test_pred <-
  knn(
    train = wbcd_train,
    test = wbcd_test,
    cl = wbcd_train_labels,
    k = 5
  )

wbcd_pred_table <- table(wbcd_test_labels, wbcd_test_pred)
wbcd_pred_table
sum(diag(wbcd_pred_table)) / nrow(wbcd_test)

# k=40
wbcd_test_pred <-
  knn(
    train = wbcd_train,
    test = wbcd_test,
    cl = wbcd_train_labels,
    k = 40
  )

wbcd_pred_table <- table(wbcd_test_labels, wbcd_test_pred)
wbcd_pred_table
sum(diag(wbcd_pred_table)) / nrow(wbcd_test)

# Q: What do our results tell us?

# Let's try a more automated approach to finding the right "k".
# We begin by creating an empty vector. 
# Hmmm. Where have we seen this before?
accuracy <- vector()

n=40

set.seed(1234)
for(k in 1:n) {
  pred <-
    knn(
      train = wbcd_train,
      test = wbcd_test,
      cl = wbcd_train_labels,
      k = k
    )
  
  pred_table <- table(wbcd_test_labels, pred)
  accuracy[k] <- sum(diag(pred_table)) / nrow(wbcd_test)
  
}

# What does our accuracy vector look like?
accuracy

# Now, we can plot the accuracy against the different values for 'k'.
tibble(value = accuracy) %>%
  ggplot(mapping=aes(x=seq(1, length(accuracy)), y=value, label=seq(1,length(accuracy)))) +
  geom_point(color="coral", size=8) +
  geom_line() +
  labs(title = "Predictive Accuracy vs. Number of Neighbors", y = "Predictive Accuracy", x = "Number of Neighbors (k)" ) +
  geom_text() +
  theme_minimal() 

# Q: What is the optimal value for 'k'?

# Note that the approach used here is rudimentary and only for illustration.
# In practice, there are packages that help with this process.
# We will get to work with one of them later in the course.

# Q: What else can we do to improve the accuracy of our model?

