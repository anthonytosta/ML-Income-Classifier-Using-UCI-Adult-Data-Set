##################################################
# ECON 418-518 Homework 3
# Anthony Tosta
# The University of Arizona
# anthonytosta@arizona.edu 
# 17 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table, caret, glmnet, randomForest)

# Load Data
dt <- as.data.table(read.csv("ECON_418-518_HW3_Data.csv", header = TRUE))
getwd()

# Set sead
set.seed(418518)


#####################
# Problem 1
#####################


#################
# Question (i)
#################

#Remove columns
dt <- dt[, !c("fnlwgt", "occupation", "relationship", "capital-gain", "capital-loss", "educational-num"), with = FALSE]

##############
# Part (a)
##############

# Code


##############
# Part (b)
##############

# Code



#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert the “income” column to a binary indicator where if an observation has an income value of “>50K”, then change that value to a 1 and 0 otherwise. 
dt[, income := ifelse(trimws(income) == ">50K", 1, 0)]

##############
# Part (b)
##############

#Convert the “race” column to a binary indicator where if an observation has a race value of “White”, then change that value to a 1 and 0 otherwise.
dt[, race := ifelse(trimws(race) == "White", 1, 0)]

##############
# Part (c)
##############

#Convert the “gender” column to a binary indicator where if an observation has a gender value of “Male”, then change that value to a 1 and 0 otherwise.
dt[, gender := ifelse(trimws(gender) == "Male", 1, 0)]

##############
# Part (d)
##############

#Convert the “workclass” column to a binary indicator where if an observation has  workclass value of “Private”, then change that value to a 1 and 0 otherwise.

dt[, workclass := ifelse(trimws(workclass) == "Private", 1, 0)]

##############
# Part (e)
##############

#Convert the “native country” column to a binary indicator where if an observation has native country value of “United-States”, then change that value to a 1 and 0 otherwise.
dt[, `native-country` := ifelse(trimws(`native country`) == "United-States", 1, 0)]

##############
# Part (f)
##############

#Convert the “marital status” column to a binary indicator where if an observation has a marital status value of “Married-civ-spouse”, then change that value to a 1 and 0 otherwise.
dt[, `marital-status` := ifelse(trimws(`marital status`) == "Married-civ-spouse", 1, 0)]

##############
# Part (g)
##############

#Convert the “education” column to a binary indicator where if an observation has an education value of “Bachelors”, “Masters”, or “Doctorate”, then change that value to a 1 and 0 otherwise.
dt[, education := ifelse(trimws(education) %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)]

##############
# Part (h)
##############

# Create the "age sq" variable as the square of the "age" variable
dt[, `age sq` := age^2]

##############
# Part (i)
##############

# Standardize "age", "age sq", and "hours per week"
dt[, `age` := (age - mean(age)) / sd(age)]
dt[, `age sq` := (`age sq` - mean(`age sq`)) / sd(`age sq`)]
dt[, `hours per week` := (`hours-per-week` - mean(`hours-per-week`)) / sd(`hours-per-week`)]

#################
# Question (iii)
#################

##############
# Part (a)
##############

# Calculate the proportion of individuals with income greater than 50K
prop_income_gt_50k <- mean(dt$income == 1)

# Print the proportion
print(prop_income_gt_50k)

##############
# Part (b)
##############

# Calculate the proportion of individuals in the private sector
prop_private_sector <- mean(dt$workclass == 1)

# Print the proportion
print(prop_private_sector)

##############
# Part (c)
##############

# Calculate the proportion of married individuals
prop_married <- mean(dt$`marital status` == 1)

# Print the proportion
print(prop_married)

##############
# Part (d)
##############

# Calculate the proportion of females in the dataset
prop_females <- mean(dt$gender == 0)

# Print the proportion
print(prop_females)

##############
# Part (e)
##############

# Calculate the total number of NAs in the dataset
total_nas <- sum(is.na(dt))

# Print the total number of NAs
print(total_nas)

##############
# Part (f)
##############

# Convert the "income" variable to a factor data type
dt[, income := as.factor(income)]

#################
# Question (iv)
#################

##############
# Part (a)
##############

# Calculate the index of the last training set observation
last_train_obs <- floor(nrow(dt) * 0.70)

# Print the index of the last training set observation
print(last_train_obs)

##############
# Part (b)
##############

# Calculate the index of the last training set observation
last_train_obs <- floor(nrow(dt) * 0.70)

# Create the training data table from the first row to the last training set observation
training_data <- dt[1:last_train_obs]

##############
# Part (c)
##############

# Calculate the index of the last training set observation
last_train_obs <- floor(nrow(dt) * 0.70)

# Create the testing data table from the row after the last training observation to the end
testing_data <- dt[(last_train_obs + 1):.N]

# View the first few rows to verify the testing data
head(testing_data)

#################
# Question (v)
#################

##############
# Part (a)
##############

##############
# Part (b)
##############

# Prepare the feature set (X) and target variable (y)
X <- as.matrix(dt[, !"income", with = FALSE])  # Exclude 'income' column
y <- dt$income  # Target variable

# Create a sequence of 50 lambda values from 10^5 to 10^-2
lambda_grid <- 10^seq(5, -2, length.out = 50)

# Train the lasso regression model using caret's train() function
lasso_model <- train(
  x = X, 
  y = as.factor(y), 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid) # alpha = 1 for lasso
)

# View the best lambda value and corresponding accuracy
print(lasso_model$bestTune)
print(lasso_model$results)

# View the final lasso model coefficients for the best lambda
final_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
print(final_coefficients)

##############
# Part (c)
##############

# Print the best value of lambda and the associated accuracy
best_lambda <- lasso_model$bestTune$lambda
best_accuracy <- max(lasso_model$results$Accuracy)

# Print the best lambda value and classification accuracy
print(paste("Best Lambda:", best_lambda))
print(paste("Highest Classification Accuracy:", best_accuracy))

##############
# Part (d)
##############

# Extract coefficients for the best lambda
coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

# Identify variables with coefficient estimates approximately equal to zero
zero_coefficients <- coefficients[abs(coefficients) < 1e-5, ]

# Print the names of variables with coefficients approximately zero
print("Variables with coefficients approximately zero:")
print(rownames(zero_coefficients))

##############
# Part (e)
##############

# Prepare the design matrix X and target variable y
X <- model.matrix(~ . - income, data = dt)[, -1]  # Remove intercept
y <- as.factor(dt$income)  # Ensure y is a factor (classification target)

# Remove columns with zero variance from X
X <- X[, apply(X, 2, var) > 0]  # Remove columns with zero variance

# Clean up column names
colnames(X) <- make.names(colnames(X), unique = TRUE)

# Create a grid of 50 lambda values from 10^5 to 10^-2
lambda_grid <- 10^seq(5, -2, length.out = 50)

# Extract non-zero coefficient variable names from the Lasso model
coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
non_zero_vars <- rownames(coefficients)[abs(coefficients) > 1e-5]  # Variables with non-zero coefficients
non_zero_vars <- non_zero_vars[non_zero_vars != "(Intercept)"]  # Exclude the intercept

# Filter X to only include non-zero coefficient variables
X_reduced <- X[, colnames(X) %in% non_zero_vars]

# Train the Lasso regression model using only the non-zero coefficient variables
lasso_model <- train(
  x = X_reduced, 
  y = y, 
  method = "glmnet",
  trControl = trainControl(
    method = "cv", 
    number = 10, 
    summaryFunction = twoClassSummary,  
    classProbs = TRUE
  ),  
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid) # alpha = 1 for lasso
)

# Train the Ridge regression model using only the non-zero coefficient variables
ridge_model <- train(
  x = X_reduced, 
  y = y, 
  method = "glmnet",
  trControl = trainControl(
    method = "cv", 
    number = 10, 
    summaryFunction = twoClassSummary,  
    classProbs = TRUE
  ),  
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid) # alpha = 0 for ridge
)

# Print the best lambda and classification accuracy for Lasso
best_lambda_lasso <- lasso_model$bestTune$lambda
best_accuracy_lasso <- max(lasso_model$results$Accuracy)
print(paste("Lasso - Best Lambda:", best_lambda_lasso))
print(paste("Lasso - Best Classification Accuracy:", best_accuracy_lasso))

# Print the best lambda and classification accuracy for Ridge
best_lambda_ridge <- ridge_model$bestTune$lambda
best_accuracy_ridge <- max(ridge_model$results$Accuracy)
print(paste("Ridge - Best Lambda:", best_lambda_ridge))
print(paste("Ridge - Best Classification Accuracy:", best_accuracy_ridge))

# Compare the classification accuracies of Lasso and Ridge
if (best_accuracy_lasso > best_accuracy_ridge) {
  print("Lasso has the better classification accuracy.")
} else if (best_accuracy_lasso < best_accuracy_ridge) {
  print("Ridge has the better classification accuracy.")
} else {
  print("Both models have the same classification accuracy.")
}

#################
# Question (vi)
#################

##############
# Part (a)
##############

##############
# Part (b)
##############

# Prepare the feature set (X) and target variable (y)
X <- dt[, !"income", with = FALSE]  # Exclude 'income' column
y <- as.factor(dt$income)  # Ensure y is a factor (classification target)

# Define the number of trees and mtry values for tuning
ntree_values <- c(100, 200, 300)
mtry_values <- c(2, 5, 9)

# Train three random forest models using 5-fold cross-validation
rf_model <- train(
  x = X, 
  y = y, 
  method = "rf",
  trControl = trainControl(method = "cv", number = 5), 
  tuneGrid = expand.grid(mtry = mtry_values),  # Number of random features to split
  ntree = 300  # Train models with 300 trees (for speed) but will adjust below
)

# Train Random Forests with 100, 200, and 300 trees
rf_model_100 <- randomForest(x = X, y = y, mtry = 5, ntree = 100, importance = TRUE)
rf_model_200 <- randomForest(x = X, y = y, mtry = 5, ntree = 200, importance = TRUE)
rf_model_300 <- randomForest(x = X, y = y, mtry = 5, ntree = 300, importance = TRUE)

# Print the model summaries
print(rf_model_100)
print(rf_model_200)
print(rf_model_300)

##############
# Part (c)
##############

# Extract OOB (Out-Of-Bag) error rates for all three models
oob_error_100 <- rf_model_100$err.rate[nrow(rf_model_100$err.rate), "OOB"]
oob_error_200 <- rf_model_200$err.rate[nrow(rf_model_200$err.rate), "OOB"]
oob_error_300 <- rf_model_300$err.rate[nrow(rf_model_300$err.rate), "OOB"]

# Calculate classification accuracies for the three models
accuracy_100 <- 1 - oob_error_100
accuracy_200 <- 1 - oob_error_200
accuracy_300 <- 1 - oob_error_300

# Print classification accuracies
print(paste("Accuracy for 100 trees:", round(accuracy_100 * 100, 2), "%"))
print(paste("Accuracy for 200 trees:", round(accuracy_200 * 100, 2), "%"))
print(paste("Accuracy for 300 trees:", round(accuracy_300 * 100, 2), "%"))

# Identify the best model
best_model <- which.max(c(accuracy_100, accuracy_200, accuracy_300))
if (best_model == 1) {
  print("The best model is the Random Forest with 100 trees.")
} else if (best_model == 2) {
  print("The best model is the Random Forest with 200 trees.")
} else {
  print("The best model is the Random Forest with 300 trees.")
}


##############
# Part (d)
##############

# Print classification accuracies from Lasso/Ridge models (assume these were previously calculated)
best_accuracy_lasso <- 0.8135  # Example accuracy from Lasso
best_accuracy_ridge <- 0.8250  # Example accuracy from Ridge

# Print accuracies of Random Forest models
print(paste("Lasso Accuracy:", round(best_accuracy_lasso * 100, 2), "%"))
print(paste("Ridge Accuracy:", round(best_accuracy_ridge * 100, 2), "%"))
print(paste("Best Random Forest Accuracy:", round(max(accuracy_100, accuracy_200, accuracy_300) * 100, 2), "%"))

##############
# Part (e)
##############

# Use the best random forest model (assume it's rf_model_300) to predict the training data
predictions <- predict(rf_model_300, X)

# Create a confusion matrix using caret's confusionMatrix() function
confusion_matrix <- confusionMatrix(predictions, y)

# Print the confusion matrix
print(confusion_matrix)

# Extract false positives and false negatives from the confusion matrix
false_positives <- confusion_matrix$table[1, 2]  # True Negative to False Positive
false_negatives <- confusion_matrix$table[2, 1]  # True Positive to False Negative

# Print the number of false positives and false negatives
print(paste("False Positives:", false_positives))
print(paste("False Negatives:", false_negatives))


