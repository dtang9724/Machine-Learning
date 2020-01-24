library(stats)
library(mice) 
library(tidyverse)
library(factoextra)

# Part I
# Import data
raw <- read_csv('https://s3.amazonaws.com/notredame.analytics.data/mallcustomers.csv')
customer <- raw

# Change the data type for each column
customer <- customer %>% 
  mutate(CustomerID = as.integer(CustomerID),Gender = as.factor(Gender),
         Age = as.integer(Age),Income = as.character(Income),SpendingScore = as.integer(SpendingScore))

# Part II
# Look at missing values
summary(customer)

# Do the imputation based on logistic regression
imputed_cust <- mice(customer,m=1,maxit=5,meth='logreg',seed=1234)

# First 10 suggested imputations of Gender
imputed_cust$imp$Gender[1:10,]

# Complete the customer dataset with the imputed values
complete_cust <- mice::complete(imputed_cust)

# Change the Income column to numbers
complete_cust$Income <-  str_remove_all(complete_cust$Income,' USD|,')
complete_cust <- complete_cust %>% 
  mutate(Income = as.numeric(Income))
summary(complete_cust)

# Part III
# Normalize the data using z-score
selected_metrics <- complete_cust %>% 
  select(-CustomerID, -Gender, -Age)
sm_z <- scale(selected_metrics)

# Cluster sm_z based on selected metrics
set.seed(1234)
k_3 <- kmeans(sm_z, centers=3, nstart = 25)
k_3$size
k_3$centers

# Viz with fviz_cluster
fviz_cluster(k_3, data = sm_z)

# Assign cluster IDs to complete_cust
complete_cust$cluster <- k_3$cluster

# Use the elbow method to identify the optimal number of clusters
wcss <- vector()
n = 20
set.seed(1234)
for(k in 1:n) {
  wcss[k] <- sum(kmeans(sm_z, k)$withinss)
}

wcss

# Visualize the values of WCSS as they relate to number of clusters
tibble(value = wcss) %>%
  ggplot(mapping=aes(x=seq(1,length(wcss)), y=value)) +
  geom_point()+
  geom_line() +
  labs(title = "The Elbow Method", y = "WCSS", x = "Number of Clusters (k)" ) +
  theme_minimal() 

# The optimal number of clusters should be 5 according to the elbow method.

# Part IV
# Use the optimal number of clusters and re-run the clustering
set.seed(1234)
k_5 <- kmeans(sm_z, centers=5, nstart = 25)
k_5$size
k_5$centers

fviz_cluster(k_5, data = sm_z)

# Looking at the above chart, we can assign the following labels to each cluster:
# Cluster 1: 'Poor spenders'
# Cluster 2: 'Neutral middle-class'
# Cluster 3: 'Rich savers'
# Cluster 4: 'Rich spenders'
# Cluster 5: 'Poor savers'

# Create a new column in complete_cust to show the cluster number
complete_cust$cluster <- k_5$cluster

# Examine age and gender for each cluster
cluster_age_gender <- complete_cust %>% 
  select(cluster,Age,Gender)

# Average age for each cluster
cluster_age_gender %>% 
  group_by(cluster) %>% 
  summarise(avg_age = mean(Age))

# Average age of the overall dataset
mean(complete_cust$Age)
# Discussion on age: we can see that the mean of age in each cluster is very different from that of the overall dataset.

# Gender distribution for each cluster with stacked barplot
cluster_gender <- cluster_age_gender %>% 
  group_by(cluster,Gender) %>% 
  tally() %>% 
  pivot_wider(names_from = Gender, values_from = n) %>% 
  as.data.frame() %>% 
  mutate(cluster = as.character(cluster))
complete_gender <- complete_cust %>% 
  group_by(Gender) %>% 
  tally() %>% 
  pivot_wider(names_from = Gender, values_from = n)
complete_gender['cluster'] <- 'Overall'
complete_gender <- complete_gender[c(3,1,2)]
gender_total <- rbind(complete_gender,cluster_gender)
gender_total <- gender_total %>% 
  transform(Femaleprob = Female/(Female+Male))
gender_total <- gender_total %>% 
  transform(Maleprob = Male/(Female+Male)) %>% 
  select(-Female,-Male) 
names(gender_total) <- c('Cluster','Female','Male')
gender_total %>% 
  pivot_longer(-Cluster,names_to = 'Gender',values_to = 'Probability') %>% 
  ggplot(aes(fill=Gender, y=Probability, x=Cluster)) + 
  geom_bar(position="stack", stat="identity")+
  geom_hline(yintercept=0.445, linetype="dashed", color = "blue")+
  theme(axis.text.x = element_text(face = c('plain', 'plain', 'plain', 'plain', 'plain', 'bold')))
# Discussion on Gender Distribition: We can see from the above graph that the overall dataset has a pretty balanced gender distribution. 
# Compared to the overall dataset, cluster 1 and 3 have the most deviation in gender distribution, where cluster 1 has significantly more females and cluster 3 more males.
# Cluster 5 also has more females but the extent is less extreme than cluster 1. Clusters 2 and 4 have similar gender distributions as in the overall dataset.

# Recommendations:
# 1. Cluster 4 would be the group to promote to because this group generally earns high income and spends frequently. This means that members in cluster 4 could be relatively price insensitive. The average age of this group is 32.7. They're likely young professionals who are financially independent. Acme can promote creative, popular, and high-quality products to this group.
# 2. Cluster 3 has high income but is conservative in spending. Acme should try to find out why these people don't spend at its stores. It could be that Acme's products don't appeal to these people. We can see that the average of of cluster 3 is 41.1. So these people can be heads of families. They might be more attracted to more family-focused products such as boats.
# 3. Cluster 5 has low income and doesn't like spending. To incentivize more spending from this group, Acme can consider strategies such as wholesale and 'everyday low price'.

