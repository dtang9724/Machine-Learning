#' ---
#' title: "K-MEANS CLUSTERING - Identifying Teen Market Segments"
#' author: "Fred Nwanganga"
#' date: "June 2nd, 2019"
#' ---
 
# We need the 'stats' package for the k-means algorithm, 
# both 'Amelia' and 'mice' for visualizing and dealing with missing values,
# factoextra for clustering and visualizing the cluster,
# and of course, the 'tidyverse' for our data exploration and preparation.
install.packages(c("stats","mice","factoextra"))

library(stats)
library(mice) # Predictive imputation - Missing values by ML
library(tidyverse)
library(factoextra)

#----------------------------------------------------------------
#' #1. Collect the Data
#----------------------------------------------------------------

teens <-
  read_csv("https://s3.amazonaws.com/notredame.analytics.data/snsdata.csv")

glimpse(teens)
head(teens)

#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------

# Get a statistical summary of the data.
summary(teens)

# Do we have any missing values in our dataset?
teens %>%
  select_if(function(x) any(is.na(x)))


# Look at the summary stats for the 'age' feature.
summary(teens$age)


# Q: What obvious problem do we see with the data?


# Since we know that the data was collected for high school students,
# we begin by eliminating the aberrant outliers in our data.
teens <- teens %>%
  mutate(age = ifelse(age >= 13 & age < 20, age, NA)) # Problem of inconsistent data


# Now what do we have?
summary(teens$age)


# Visualize the distribution for the 'age' feature (excluding NAs).
teens %>%
  filter(!is.na(age)) %>%
  ggplot() +
  geom_histogram(mapping=aes(age), fill="white", color="black")


# Let's use mean imputation by graduation year to resolve the missing age values.
teens <- teens %>%
  group_by(gradyear) %>%
  mutate(age = ifelse(!is.na(age), age, mean(age, na.rm=TRUE)))

# Mean imputation
# Group by gradyear is hot-deck imputation.


# What does our data look like?
summary(teens$age)


# Now, let's deal with the missing values for the 'gender' feature.
# We will use the predictive imputation approach to fill in the missing values.
# The 'mice' package provides us with a toolset for this and more.


# Start by converting gender to a factor.
teens$gender <- as.factor(teens$gender)


summary(teens$gender)


# Generate dataset of suggested imputations.
# The mice() function takes a few arguments. We specify 'm', 'maxit', 'meth' and 'seed'.
# m: specifies the number of candidate predictions to make for each instance.
# maxit: specifies the number of iterations the function goes through for each prediction.
# meth: specifies the type of algorithm to use for the prediction. Here we use logistic regression.
# seed: specifies a random seed number. You can set this to whatever you like.
imputed_teens <- mice(teens,m=1,maxit=5,meth='logreg',seed=1234)


# What are the first 10 suggested imputations?
imputed_teens$imp$gender[1:10,]


# Create new dataset with missing values replaced with suggestions.
# We choose the second(2) set of suggested imputations.
complete_teens <- mice::complete(imputed_teens)


# What does our gender feature now look like?
summary(complete_teens$gender)


# And how does its distribution compare with the original dataset?
round(prop.table(table(teens$gender)),3)
round(prop.table(table(complete_teens$gender)),3)


# We're done dealing with the missing values in our dataset. 
# It's time to move on to data preparation.
# We only need the 36 student interest features for clustering.
# Let's create a new dataset with only the features we need. 
interests <- complete_teens %>%
  select(-gradyear, -gender, -age, -friends)


summary(interests)

# The summary statistics show a wide range of values for the interest counts.
# In order to avoid features with large ranges from dominating our model,
# we need to normalize the features using the scale() function for z-score normalization.
interests_z <- scale(interests)

summary(interests_z)

#----------------------------------------------------------------
#' #3. "Train" the Model
#----------------------------------------------------------------

# We are now ready to attempt to cluster the data.
# We set the value for k to 5 and choose to use 25 different initial configurations.
set.seed(1234)
k_3 <- kmeans(interests_z, centers=3, nstart = 25)# do this 25 times.

#----------------------------------------------------------------
#' #4. "Evaluate" the Model's Performance
#----------------------------------------------------------------

# One way to evaluate the quality of clustering is the size of the clusters.
# If the clusters are too large or too small compared to others, they are unlikely to be useful.


# Let's take a look at the size of the clusters.
k_3$size


# Cluster 2 looks a bit suspect. We'll keep an eye on it.


# ...and the cluster centers.
k_3$centers #Try to print this in a list.


# Q: How do we interpret these numbers?


# We can also visualize the clusters to get additional insight.
fviz_cluster(k_3, data = interests_z)



# Apply the cluster IDs to the original data frame for further evaluation
complete_teens$cluster <- k_3$cluster


# Now, let's take a look at the first ten records for some of the features
complete_teens %>%
  select(cluster, gender, age, friends) %>%
  slice(1:10)


# To further evaluate our results, we need to look at how 
# the clusters correlate with what we know about the students.


# Look at mean age by cluster. What does it tell us?
complete_teens %>%
  summarize(age = mean(age))

complete_teens %>%
  group_by(cluster) %>%
  summarize(age = mean(age))


# Look at gender counts by cluster.
# We need to create male and female dummy variables for this.
complete_teens <- complete_teens %>%
  mutate(female = ifelse(gender == 'F',1,0)) %>%
  mutate(male = ifelse(gender == 'M',1,0))


# Let's look the female population. What do we learn?
complete_teens %>%
  summarize(female = mean(female))

complete_teens %>%
  group_by(cluster) %>%
  summarize(female = mean(female))
# Preportion of female and males

# What about for males?
complete_teens %>%
  summarize(male = mean(male))

complete_teens %>%
  group_by(cluster) %>%
  summarize(male = mean(male))


# Look at mean number of friends by cluster. What does it tell us?
complete_teens %>%
  summarize(friends = mean(friends))

complete_teens %>%
  group_by(cluster) %>%
  summarize(friends = mean(friends))


#----------------------------------------------------------------
#' #5. "Improve" the Model's Performance
#----------------------------------------------------------------

# Let's see how varying the number of clusters affects the results.
k_4 <- kmeans(interests_z, centers = 4, nstart = 25)
k_5 <- kmeans(interests_z, centers = 5, nstart = 25)
k_6 <- kmeans(interests_z, centers = 6, nstart = 25)

# Plot and compare the results.
p1 <- fviz_cluster(k_3, geom = "point", data = interests_z) + ggtitle("k = 3")
p2 <- fviz_cluster(k_4, geom = "point",  data = interests_z) + ggtitle("k = 4")
p3 <- fviz_cluster(k_5, geom = "point",  data = interests_z) + ggtitle("k = 5")
p4 <- fviz_cluster(k_6, geom = "point",  data = interests_z) + ggtitle("k = 6")

# Here, we make use of the grid.arrange() function to display several plots at the same time.
# This function is part of the griExtra package. Install it, if you haven't already.
library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)


# Q: Can you tell what the ideal number of clusters should be from the visuals?


# Now let's try to choose an ideal value for k based on the elbow method.
# The first thing we do is, create an empty vector to house the values for WCSS,
wcss <- vector()


# ... then specify the loop that generates the values.
n = 20
set.seed(1234)
for(k in 1:n) {
  wcss[k] <- sum(kmeans(interests_z, k)$withinss)
}

wcss

# Visualize the values of WCSS as they relate to number of clusters
tibble(value = wcss) %>%
  ggplot(mapping=aes(x=seq(1,length(wcss)), y=value)) +
  geom_point()+
  geom_line() +
  labs(title = "The Elbow Method", y = "WCSS", x = "Number of Clusters (k)" ) +
  theme_minimal() 


# Q: What is the optimal value for k?
k_7 <- kmeans(interests_z, centers = 7, nstart = 25)
# Write out each clusters.
k_7$centers
fviz_cluster(k_7, geom = "point",  data = interests_z) + ggtitle("k = 7")

