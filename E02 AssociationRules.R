#' ---
#' title: "ASSOCIATION RULES - Identifying Frequently-Purchased Groceries"
#' author: "Fred Nwanganga"
#' date: "June 2nd, 2019"
#' ---

# Install and load the association rules (arules) package.
# install.packages("arules")
# install.packages("tidyverse")
library(arules)
library(tidyverse)


#----------------------------------------------------------------
#' #1. Collect the Data
#----------------------------------------------------------------

# Load the grocery data into a sparse matrix (Note the use of 'read.transactions' for this dataset).
groceries <-
  read.transactions("https://s3.amazonaws.com/notredame.analytics.data/groceries.csv",
                    sep = ",")


#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------

# Let's take a look at the summary stats for the dataset.
summary(groceries)


# Q:What does the density value mean?
# It means the percentage of non-empty cells in our sparse matrix.
# In other words, it is the number of total number of purchased items divided by all items.

# Q:What do the other summary statistics tell us?
# 2159 people bought 1 item, etc. 

# The entire sparse matrix can also be visualized using the image() function.
# Let's visualize a random sampling of the data.
image(sample(groceries, 100))


# Q:What does this visualization further illustrate?


# Let's take a look at the first five transactions.
# Note that for a sparse matrix the arules package provides the inspect() function.
inspect(groceries[1:5])


# Using itemFrequency(), we can examine the frequency of the first five items (in alphabetical order).
itemFrequency(groceries[, 1:5])


# We can also visualize the item frequency data.
# To do this, we use the itemFrequencyPlot() function.
# Let's look at a histogram of items with at least 10 percent support.
itemFrequencyPlot(groceries, support = 0.1)


# Show histogram of the top 20 items in decreasing order of support.
itemFrequencyPlot(groceries, topN = 20)


# The arules package provides a limited number of functions for exploring the data.
# To do further data exploration, we can create a dataframe of the items and their purchase frequency.

# Let's get a view of the frequency of all the items in our data set.
itemFrequency(groceries)

# Now, we can convert the data to a data frame.
groceries.frequency <-
  data.frame(
    Items = names(itemFrequency(groceries)),
    Frequency = itemFrequency(groceries),
    row.names = NULL
  )

# What do we have?
head(groceries.frequency)


# Q: What are the 10 most frequently bought items at this store?
groceries.frequency %>%
  arrange(desc(Frequency)) %>%
  slice(1:10)

# We can plot this data as well using ggplot.
groceries.frequency %>%
  arrange(desc(Frequency)) %>%
  slice(1:20) %>%
  ggplot() +
  geom_col(mapping=aes(x=reorder(Items,Frequency), y=Frequency), fill="lightblue", color="black") +
  labs(x="Items") +
  coord_flip() + theme_minimal() + theme(legend.position = "none")


# Q: What are the 10 least frequently bought items at this store?
groceries.frequency %>%
  arrange(Frequency) %>%
  slice(1:10)


#----------------------------------------------------------------
#' #3. "Train" the Model
#----------------------------------------------------------------


# Using the default settings for the apriori() function, we attempt to generate some rules.
# The default settings are: support = 0.1, confidence = 0.8 and minimum required rule items = 1.
groceryrules <- apriori(groceries)


# What do we have?
groceryrules


# Q:Why did we get the result that we did?


# With the minimum support threshold set at 0.1, in order for an item to be part of a rule,
# the item must have appeared in at least 983.5 (0.1 * 9835) transactions.

# These are the items that meet that requirement. 
itemFrequencyPlot(groceries, support = 0.1)

# Unfortunately, none of the rules for these items have a confidence threshold at or above 0.8.

# We need to relax our thresholds a bit to get some rules from our data.
# Let's say we decide to include items that were purchased on average at least 5 times a day.
# Given that our data is for 30 days, we would need to set our support threshold at 0.015 -> ((5*30)/9835).


# A minimum confidence threshold of 0.8 implies that the rule has to be correct at least 80% of the time. 
# This is rather strict requirement for our data. Let's set that at 25% for now.


# Now we make another attempt at generating rules (using the 'parameter' attribute to set our thresholds).
groceryrules <-
  apriori(groceries,
          parameter = list(
            support = 0.015,
            confidence = 0.25,
            minlen = 2
          ))


groceryrules


# Houston, we have lift-off!!!!


#----------------------------------------------------------------
#' #4. "Evaluate" the Model's Performance
#----------------------------------------------------------------


# Summary of the grocery association rules.
summary(groceryrules)


# Q:What do the summary stats mean?


# Let's take a look at the first 10 rules.
inspect(groceryrules[1:10])


# Q: We learned that Association rules fall into 3 categories: "Actionable", "Trivial" and "Inexplicable". 
# Based on this, how would you categorize our first 10 rules?


#----------------------------------------------------------------
#' #5. "Improve" the Model's Performance
#----------------------------------------------------------------

# In practice, we usually will have hundreds/thousands of rules generated from our data.
# Since we cannot (or should not) manually parse through hundreds of rules,
# we need to find ways to identify the rules that may be useful to us.


# Let's start by sorting the grocery rules by lift and examining the top 5.
groceryrules %>%
  sort(by = "lift") %>%
  head(n=5) %>%
  inspect() 



# Q: What does the lift value mean for our third rule?


# Suppose we want to figure out whether 'tropical fruit' are also often purchased with other items. 
# We would use the subset() function to find subsets of rules containing any 'tropical fruit' items.
groceryrules %>%
  subset(items %in% "tropical fruit") %>%
  inspect()


# And, if we wanted to see the top 5 rules in terms of lift that have "tropical fruit" in the LHS?
# We would combine both the sort() and subset() functions to do this.
groceryrules %>%
  subset(lhs %in% "tropical fruit") %>%
  sort( by = "lift") %>%
  head(n=5) %>%
  inspect()


# Q: What are some of the actions we can take in response to our results?


# Note that the subset() function can be used with several keywords and operators:
# - The keyword 'items', matches an item **appearing anywhere** in the rule.
# - Limit the subset with 'lhs' and 'rhs' instead.
# - The operator %in% means that at least one of the items must be found in the list you defined.
# - For partial matching (%pin%) and complete matching (%ain%).
# - We can also filter by support, confidence, or lift.
# - We can also combine standard R logical operators such as and (&), or (|), and not (!).


# Finally, we can output our rules to a CSV file for additional analysis in a separate platform,...
write(
  groceryrules,
  file = "groceryrules.csv",
  sep = ",",
  quote = TRUE,
  row.names = FALSE
)


# ... or we can convert it to a data frame for further analysis in R.
groceryrules.data <- as(groceryrules, "data.frame")
head(groceryrules.data)






