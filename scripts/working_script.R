##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##################################################
#######################ANALYSIS###################
##################################################


#look at the structure
str(edx)

#check for null values
colSums(is.na(edx))
#there are no null values
#look at summary statistics of the ratings
summary(edx$rating)
#produce a histogram of the ratings
ggplot(edx) + geom_histogram(aes(rating), color = "black")

# Check the frequency of movies appearing in the database
edx %>% count(movieId) %>% ggplot() + 
  geom_histogram(aes(n), bins = 30, color = "black") + 
  scale_x_log10()

# Check the frequency of users appearing in the database
edx %>% count(userId) %>% ggplot() + 
  geom_histogram(aes(n), bins = 30, color = "black") + 
  scale_x_log10()

# Check the frequency with which genres appear in the database
edx %>% count(genres) %>% ggplot() + 
  geom_histogram(aes(n)) + scale_x_log10()
##from all the above, we see that there is variation. Some
##movies appear a few times while some apear a lot. Same applies
##to users and genres. This means that we will probably use
## regularisation to penalise records

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate the overall mean
mu <- mean(edx$rating)

# Use the mean to predict
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse
rmse_results <- data.frame(method = "Just the average", RMSE = naive_rmse)

# Genres effects
genres_avgs <- edx %>% group_by(genres) %>% summarize(b_g = mean(rating - mu))
predicted_ratings <- mu + validation %>% 
  left_join(genres_avg, by = "genres") %>% 
  pull(b_g)
genres_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- rmse_results %>% 
  add_row(method = "Genres effect", RMSE = genres_rmse)

# Movie effects
movie_avgs <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  pull(b_i)
movie_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- rmse_results %>% 
  add_row(method = "Movie effect", RMSE = movie_rmse)

#User effects
user_avgs <- edx %>% group_by(userId) %>% summarize(b_u = mean(rating - mu))
predicted_ratings <- mu + validation %>% 
  left_join(user_avgs, by = "userId") %>% 
  pull(b_u)
user_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- rmse_results %>% 
  add_row(method = "User effect", RMSE = user_rmse)

# User and movie effects together
user_avgs <- edx %>% left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% summarise(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% pull(pred)
usermovie_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- rmse_results %>% 
  add_row(method = "User and movie effect", RMSE = usermovie_rmse)

# Regularization
#Penalise movies, user, and genres with a small 
#number of records

#Use cross-validation to choose lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  b_i <- edx %>% group_by(movieId) %>% 
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx %>% left_join(b_i, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u = sum(rating - b_i - mu)/(n() + l))
  
  b_g <- edx %>% left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% group_by(genres) %>% 
    summarise(b_g = sum(rating - b_i - b_u - mu)/(n() + l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(pred = mu + b_i + b_u + b_g) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
rmse_results <- rmse_results %>% 
  add_row(method = "User, movie, and genres reg.", RMSE = min(rmses))

# Save the coefficients from the optimal lambda
b_i <- edx %>% group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu)/(n() + lambda))

b_u <- edx %>% left_join(b_i, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u = sum(rating - b_i - mu)/(n() + lambda))

b_g <- edx %>% left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% group_by(genres) %>% 
  summarise(b_g = sum(rating - b_i - b_u - mu)/(n() + lambda))

# Working on the date
library(lubridate)
edx %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(week = week(date)) %>% group_by(week) %>% 
  summarise(rating = mean(rating)) %>% ggplot() + 
  geom_point(aes(week, rating))

b_w <- edx %>% left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by = "genres") %>% 
  mutate(date = as_datetime(timestamp)) %>%
  mutate(week = week(date)) %>% group_by(week) %>% 
  summarise(b_w = mean(rating - b_g - b_u - b_i - mu))

predicted_ratings <- validation %>%
  mutate(date = as_datetime(timestamp)) %>%
  mutate(week = week(date)) %>% 
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by = "genres") %>% 
  left_join(b_w, by = "week") %>% 
  mutate(pred = mu + b_i + b_u + b_g + b_w) %>% 
  pull(pred)
rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- rmse_results %>% 
  add_row(method = "User, movie, genres reg. with date", RMSE = rmse)
rmse_results
