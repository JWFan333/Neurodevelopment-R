
library(caret)    # Machine learning framework
library(pROC)     # ROC curve analysis
library(ggplot2)  # Visualization
library(shapviz)  # SHAP analysis
library(kernelshap)
library(dplyr)    # Data manipulation
library(rmda)
library(catboost)
library(lightgbm)
library(DALEX)

# ===============================================
# User configuration
# ===============================================
data_dir <- "data"          # Input folder containing train_data.csv and test_data.csv
output_dir <- "output"      # Output folder for all generated files
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ===============================================
# 1. Data loading
train_data <- read.csv(file.path(data_dir, "train_data.csv"), header = TRUE, row.names = 1)
test_data <- read.csv(file.path(data_dir, "test_data.csv"), header = TRUE, row.names = 1)

# Move to output directory so all generated files are saved there
setwd(output_dir)

train_group <- data.frame(Group = substr(rownames(train_data), 1, 2), row.names = rownames(train_data))
test_group <- substr(rownames(test_data), 1, 2)
test_group  <- ifelse(test_group  == 'HC', '0','1')

train_data$Group <- as.factor(train_group$Group)

table(train_data$Group)
table(test_group)
set.seed(12345)

# 2. Specify the response variable name. Keep "Group" consistent to avoid many downstream changes.
response_var <- "Group"

# 3. Set cross-validation parameters. Use fewer folds for smaller samples to avoid errors.
cv_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

# === Performance metric calculation function ==========================================
calculate_metrics <- function(truth, probs, threshold = 0.5) {
  # Convert probabilities to predicted labels
  pred <- factor(ifelse(probs >= threshold, "Positive", "Negative"), 
                 levels = c("Negative", "Positive"))
  
  # Convert true labels to factors
  truth_fct <- factor(ifelse(truth == 1, "Positive", "Negative"), 
                      levels = c("Negative", "Positive"))
  
  # Compute confusion matrix
  cm <- confusionMatrix(pred, truth_fct, positive = "Positive")
  
  # Extract metrics
  metrics <- c(
    cm$byClass["Sensitivity"],  # Sensitivity = TP/(TP+FN)
    cm$byClass["Specificity"],  # Specificity = TN/(TN+FP)
    cm$overall["Accuracy"],     # Accuracy = (TP+TN)/(TP+FP+TN+FN)
    cm$byClass["Pos Pred Value"],     # Positive predictive value = TP/(TP+FP)
    cm$byClass["Neg Pred Value"],     # Negative predictive value = TN/(TN+FN)
    cm$byClass["F1"]     
  )
  metrics["Youden"] <- metrics["Sensitivity"] + metrics["Specificity"] - 1  # Youden index
  return(metrics)
}
# ===============================================================

# Validate data format
if(!response_var %in% colnames(train_data)) {
  stop(paste("Response variable", response_var, "does not exist in training data"))
}

# Ensure response variable is a factor
train_data[[response_var]] <- factor(train_data[[response_var]], 
                                     levels = c("HC", "SZ"))
formula <- as.formula(paste(response_var, "~ ."))

## List of 13 models (XGBoost added)
model_settings <- data.frame(
  AlgorithmName = c("RandomForest", "GradientBoosting", "SVM_Kernel",
                    "LogisticModel", "NeighborMethod","PLSModel", 
                    "BoostingMethod", "NeuralNet", "BayesMethod", 
                    "DiscriminantModel",'Lasso',"AdaptiveBoosting", "XGBoost"),
  Implementation = c("rf", "xgbTree", "svmRadial", 
                     "glm", "knn","pls", 
                     "gbm", "nnet", "nb", 
                     "lda",'glmnet',"AdaBoost.M1", "xgbLinear")
)

# Note: AdaptiveBoosting may be slow and can take several minutes depending on hardware.
# You may skip it if needed.
# "AdaptiveBoosting"
# "AdaBoost.M1"

# Create result storage lists

# Store model performance metrics:
train_metrics <- list()
test_metrics <- list()

# Store model ROC objects:
train_roc_list <- list()  
test_roc_list <- list() 

modelContainer <- list()
AUCresults <- c()
training_times <- numeric()

# Create lists to store predicted probabilities for DCA
train_probs_list <- list()
test_probs_list <- list()

# Add residual calculation lists in the model training loop
residuals_train_list <- list()
residuals_test_list <- list()

# Updated 2025-07-30: add a list to record grid hyperparameters.
best_tune_params <- list()
# Start total timer
total_start <- Sys.time()

cat("===== Start model training ", format(total_start, "%Y-%m-%d %H:%M:%S"), " =====\n")

# Model training and evaluation
for (idx in seq_len(nrow(model_settings))) {
  algoName <- model_settings$AlgorithmName[idx]
  algoImpl <- model_settings$Implementation[idx]
  
  # Start model timer
  start_time <- Sys.time()
  cat(sprintf("\n[%d/%d] Start training %s: %s\n", idx, nrow(model_settings), algoName, format(start_time, "%H:%M:%S")))
  
  tryCatch({
    # Notes for special models
    slow_models <- c("SVM_Kernel", "NeuralNet", "GradientBoosting", "AdaptiveBoosting", "XGBoost")
    if (algoName %in% slow_models) {
      cat("  ! Note: this model may take a long time...\n")
    }
    
    # Model training
    cat("  - Start model training...\n")
    
    # Apply algorithm-specific settings
    if (algoName == "SVM_Kernel") {
      # SVM tuning
      svm_tuneGrid <- expand.grid(
        sigma = c(0.01, 0.1, 1),
        C = c(0.25, 0.5, 1, 2)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   prob.model = TRUE, trControl = cv_control,
                                   tuneGrid = svm_tuneGrid)
    } else if (algoName == "GradientBoosting") {
      # XGBoost tuning
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1,
                                                          gamma=0, colsample_bytree=0.7,
                                                          min_child_weight=1, subsample=0.7))
    } else if (algoName == "RandomForest") {
      # Random forest tuning
      rf_tuneGrid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = rf_tuneGrid)
    } else if (algoName == "NeighborMethod") {
      # KNN tuning parameters
      knn_tuneGrid <- expand.grid(k = c(3, 5, 7, 9, 11, 13, 15))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = knn_tuneGrid)
    } else if (algoName == "PLSModel") {
      # PLS tuning parameters
      pls_tuneGrid <- expand.grid(ncomp = c(1, 2, 3, 4, 5))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = pls_tuneGrid)
    } else if (algoName == "BoostingMethod") {
      # GBM tuning parameters
      gbm_tuneGrid <- expand.grid(
        n.trees = c(50, 100, 150),
        interaction.depth = c(1, 3, 5),
        shrinkage = c(0.01, 0.1),
        n.minobsinnode = c(5, 10)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = gbm_tuneGrid)
    } else if (algoName == "NeuralNet") {
      # Neural network tuning parameters
      nnet_tuneGrid <- expand.grid(
        size = c(3, 5, 7),
        decay = c(0, 0.001, 0.01)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = nnet_tuneGrid)
    } else if (algoName == "BayesMethod") {
      # Naive Bayes tuning parameters
      nb_tuneGrid <- expand.grid(
        fL = c(0, 0.5, 1),
        usekernel = c(TRUE, FALSE),
        adjust = c(0.5, 1, 1.5)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = nb_tuneGrid)
    } else if (algoName == "Lasso") {
      # Lasso/elastic net tuning parameters
      glmnet_tuneGrid <- expand.grid(
        alpha = c(0, 0.5, 1),  # 0=ridge, 1=lasso, 0.5=elastic net
        lambda = c(0.001, 0.01, 0.1, 1)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = glmnet_tuneGrid)
    } else if (algoName == "AdaptiveBoosting") {
      # AdaBoost tuning parameters
      adaboost_tuneGrid <- expand.grid(
        mfinal = c(50, 100, 150),  # Number of trees
        maxdepth = c(1, 3, 5),     # Maximum tree depth
        coeflearn = "Breiman"      # Coefficient learning algorithm
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = adaboost_tuneGrid)
    } else if (algoName == "XGBoost") {
      # XGBoost linear model tuning parameters
      xgb_tuneGrid <- expand.grid(
        nrounds = c(50, 100, 150),  # Boosting iterations
        lambda = c(0, 0.1, 1),      # L2 regularization
        alpha = c(0, 0.1, 1),       # L1 regularization
        eta = c(0.01, 0.1, 0.3)     # Learning rate
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = xgb_tuneGrid)
    } else {
      # For models without tuning parameters (e.g., glm, lda), run with defaults
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control)
    }
    
    # Calculate training time
    model_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  - Training completed (elapsed: %s sec)\n", model_time))
    
    # +++ Extract and print best parameters +++
    cat("  - Best parameter combination:\n")
    print(trainedModel$bestTune)
    
    # Store best parameters
    best_tune_params[[algoName]] <- trainedModel$bestTune
    
    # Add evaluation information
    cat("  - Start model evaluation...\n")
    
    # === Training set performance metrics =========================================
    # Get cross-validation predicted probabilities
    cv_probs <- trainedModel$pred[trainedModel$pred$rowIndex, "SZ"]
    cv_indices <- trainedModel$pred[trainedModel$pred$rowIndex, "rowIndex"]
    
    # Extract true labels
    truth_train <- ifelse(train_data[[response_var]][cv_indices] == "SZ", 1, 0)
    
    # Calculate training set metrics
    train_metrics[[algoName]] <- calculate_metrics(truth_train, cv_probs)
    
    # Create training set ROC object
    roc_train <- roc(truth_train, cv_probs)
    train_roc_list[[algoName]] <- roc_train
    
    # === Test set performance metrics =========================================
    test_probs <- predict(trainedModel, newdata = test_data, type = "prob")[, "SZ"]
    
    # Calculate test set metrics
    test_metrics[[algoName]] <- calculate_metrics(test_group, test_probs)
    
    # Create test set ROC object
    roc_test <- roc(test_group, test_probs)
    test_roc_list[[algoName]] <- roc_test
    
    # Store model/AUC results
    modelContainer[[algoImpl]] <- trainedModel
    AUCresults <- c(AUCresults, paste0(algoName, ": ", 
                                       sprintf("%.03f", roc_test$auc)))
    
    # Store predicted probabilities for DCA
    # Training set: create full probability vector
    train_probs_full <- rep(NA, nrow(train_data))
    train_probs_full[trainedModel$pred$rowIndex] <- trainedModel$pred$DN
    train_probs_list[[algoName]] <- train_probs_full
    
    # Test set: store directly
    test_probs_list[[algoName]] <- test_probs
    
    # Performance output
    cat(sprintf("  - Training AUC: %.3f\n", auc(roc_train)))
    cat(sprintf("  - Test AUC: %.3f\n", auc(roc_test)))
    
    # Record training time
    training_times[algoName] <- model_time
    
  }, error = function(e) {
    # Record error message and elapsed time
    error_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  ! Error: %s (elapsed: %s sec)\n", e$message, error_time))
    
    # Record error information
    train_metrics[[algoName]] <- rep(NA, 7)
    test_metrics[[algoName]] <- rep(NA, 7)
    training_times[algoName] <- error_time
    # Store NA to indicate parameter extraction failed
    best_tune_params[[algoName]] <- NA
  })
  
  # Show single-model completion status
  cat(sprintf("[%d/%d] Completed %s | cumulative elapsed: %.1f sec\n", idx, nrow(model_settings), algoName, 
              as.numeric(difftime(Sys.time(), total_start, units = "secs"))))
  
  # Add separator
  cat("-----------------------------------------\n")
}
# Final report
total_time <- round(as.numeric(difftime(Sys.time(), total_start, units = "mins")), 1)
cat("\n===== Training of 13 models completed! Total elapsed: ", total_time, "minutes =====\n")

# Two special models need to be trained separately.
# === After the model loop, train CatBoost separately ========================
cat("\n===== Start training CatBoost separately =====\n")

# 1. Prepare data format required by CatBoost
train_features <- train_data[, -which(names(train_data) == response_var)]
train_labels <- ifelse(train_data[[response_var]] == "SZ", 1, 0)

test_features <- test_data
test_labels <- test_group

# Create CatBoost Pool objects
train_pool <- catboost.load_pool(data = train_features, label = train_labels)
test_pool <- catboost.load_pool(data = test_features, label = as.numeric(test_labels))

# 2. Define parameter grid
catboost_grid <- expand.grid(
  depth = c(4, 6, 8),          # Tree depth
  learning_rate = c(0.01, 0.05, 0.1),  # Learning rate
  l2_leaf_reg = c(1, 3, 5)     # L2 regularization coefficient
)

# Initialize best parameters and performance
best_auc <- 0
best_params <- NULL
best_iter <- NULL

# 3. Grid search
cat("-- Start grid search for best parameters --\n")
for (i in 1:nrow(catboost_grid)) {
  params <- as.list(catboost_grid[i, ])
  
  # Set basic parameters
  cv_params <- list(
    iterations = 500,
    learning_rate = params$learning_rate,
    depth = params$depth,
    l2_leaf_reg = params$l2_leaf_reg,
    loss_function = 'Logloss',
    eval_metric = 'AUC',
    early_stopping_rounds = 50,
    random_seed = 42
  )
  
  # Run cross-validation
  cv_results <- catboost.cv(
    pool = train_pool,
    params = cv_params,
    fold_count = 5,
    partition_random_seed = 42,
    shuffle = TRUE
  )
  
  # Get best iteration and AUC
  iter <- which.max(cv_results$test.AUC.mean)
  auc_value <- max(cv_results$test.AUC.mean)
  
  cat(sprintf("Parameter combination %d/%d: depth=%d, lr=%.3f, l2=%.1f | best iteration: %d, AUC: %.4f\n",
              i, nrow(catboost_grid), params$depth, params$learning_rate, 
              params$l2_leaf_reg, iter, auc_value))
  
  # Update best parameters
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- cv_params
    best_iter <- iter
  }
}

# 4. Train final model with best parameters
cat(sprintf("\nBest parameters: depth=%d, lr=%.3f, l2=%.1f | best iteration: %d, AUC: %.4f\n",
            best_params$depth, best_params$learning_rate, 
            best_params$l2_leaf_reg, best_iter, best_auc))
# Add CatBoost best parameters to summary list
catboost_best_params <- data.frame(
  depth = best_params$depth,
  learning_rate = best_params$learning_rate,
  l2_leaf_reg = best_params$l2_leaf_reg,
  iterations = best_iter
)
best_tune_params[["CATBoost"]] <- catboost_best_params

# Set final model parameters
# Updated 2025-09-11: add best hyperparameters here
catboost_params <- best_params
catboost_params$iterations <- best_iter
catboost_params$depth <- best_params$depth
catboost_params$learning_rate <- best_params$learning_rate
catboost_params$l2_leaf_reg <- best_params$l2_leaf_reg

# 5. Train CatBoost model and time it
start_time_catboost <- Sys.time()
cat(sprintf("Start training CatBoost: %s\n", format(start_time_catboost, "%H:%M:%S")))

catboost_model <- catboost.train(
  learn_pool = train_pool,
  test_pool = test_pool,
  params = catboost_params
)

# Calculate training time
catboost_train_time <- round(as.numeric(difftime(Sys.time(), start_time_catboost, units = "secs")), 1)
cat(sprintf("CatBoost training completed (elapsed: %s sec)\n", catboost_train_time))

# 4. Get predicted probabilities
# Training set predicted probabilities
train_catboost_preds <- catboost.predict(catboost_model, train_pool, prediction_type = "Probability")
# Test set predicted probabilities
test_catboost_preds <- catboost.predict(catboost_model, test_pool, prediction_type = "Probability")

# 5. Calculate performance metrics and integrate into result lists
algoName <- "CATBoost"

# Training set metrics
train_truth <- ifelse(train_data[[response_var]] == "SZ", 1, 0)
train_metrics[[algoName]] <- calculate_metrics(train_truth, train_catboost_preds)

# Test set metrics
test_metrics[[algoName]] <- calculate_metrics(as.numeric(test_labels), test_catboost_preds)

# Create ROC objects
roc_train_catboost <- roc(train_truth, train_catboost_preds)
roc_test_catboost <- roc(as.numeric(test_labels), test_catboost_preds)

# Add to ROC list
train_roc_list[[algoName]] <- roc_train_catboost
test_roc_list[[algoName]] <- roc_test_catboost

# Add to model container and AUC results
modelContainer[["catboost"]] <- catboost_model
AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_catboost$auc)))

# Add to probability list (for DCA)
train_probs_list[[algoName]] <- train_catboost_preds
test_probs_list[[algoName]] <- test_catboost_preds

# === Add CatBoost residual calculation ===
# Training residuals = actual values - predicted probabilities
train_residuals_catboost <- train_truth - train_catboost_preds
# Test residuals = actual values - predicted probabilities
test_residuals_catboost <- as.numeric(test_labels) - test_catboost_preds

# Store results
residuals_train_list[[algoName]] <- train_residuals_catboost
residuals_test_list[[algoName]] <- test_residuals_catboost

# Record training time
training_times[algoName] <- catboost_train_time

# Performance output
cat(sprintf("  - CatBoost training AUC: %.3f\n", auc(roc_train_catboost)))
cat(sprintf("  - CatBoost test AUC: %.3f\n", auc(roc_test_catboost)))

# === After the model loop, train LightGBM separately ======================
cat("\n===== Start training LightGBM separately =====\n")
# 1. Prepare data format required by LightGBM
train_features_lgb <- train_data[, -which(names(train_data) == response_var)]
train_labels_lgb <- ifelse(train_data[[response_var]] == "SZ", 1, 0)

test_features_lgb <- test_data
test_labels_lgb <- as.numeric(test_group)

# Handle categorical variables
categorical_cols <- names(train_features_lgb)[sapply(train_features_lgb, is.factor)]

# Convert to matrix format
train_matrix_lgb <- as.matrix(train_features_lgb)
test_matrix_lgb <- as.matrix(test_features_lgb)

# 2. Create LightGBM datasets
dtrain_lgb <- lgb.Dataset(
  data = train_matrix_lgb,
  label = train_labels_lgb,
  categorical_feature = categorical_cols
)

dtest_lgb <- lgb.Dataset(
  data = test_matrix_lgb,
  label = test_labels_lgb,
  reference = dtrain_lgb,
  categorical_feature = categorical_cols
)

# 3. Define parameter grid
lgb_grid <- expand.grid(
  num_leaves = c(15, 31, 63),          # Number of leaves
  learning_rate = c(0.01, 0.05, 0.1),  # Learning rate
  min_data_in_leaf = c(5, 10, 20),     # Minimum data in leaf
  lambda_l2 = c(0, 0.1, 0.5),          # L2 regularization coefficient
  feature_pre_filter= F
)

# Initialize best parameters and performance
best_auc <- 0
best_params <- NULL
best_iter <- NULL

# 4. Grid search
cat("-- Start grid search for best parameters --\n")
for (i in 1:nrow(lgb_grid)) {
  params <- as.list(lgb_grid[i, ])
  
  lgb_params <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = params$num_leaves,
    learning_rate = params$learning_rate,
    min_data_in_leaf = params$min_data_in_leaf,
    lambda_l2 = params$lambda_l2,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 5,
    verbosity = -1,
    seed = 42,
    feature_pre_filter = FALSE  # Key setting here
  )
  
  # Cross-validation
  cv_model <- lgb.cv(
    params = lgb_params,
    data = dtrain_lgb,
    nrounds = 500,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 50,
    eval_freq = 10
  )
  
  # Get best iteration and AUC
  iter <- cv_model$best_iter
  auc_value <- max(unlist(cv_model$record_evals$valid$auc$eval))
  
  cat(sprintf("Parameter combination %d/%d: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | best iteration: %d, AUC: %.4f\n",
              i, nrow(lgb_grid), params$num_leaves, params$learning_rate, 
              params$min_data_in_leaf, params$lambda_l2, iter, auc_value))
  
  # Update best parameters
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- lgb_params
    best_iter <- iter
  }
}

# 5. Train final model with best parameters
cat(sprintf("\nBest parameters: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | best iteration: %d, AUC: %.4f\n",
            best_params$num_leaves, best_params$learning_rate, 
            best_params$min_data_in_leaf, best_params$lambda_l2, best_iter, best_auc))

# Add LightGBM best parameters to summary list
lightgbm_best_params <- data.frame(
  num_leaves = best_params$num_leaves,
  learning_rate = best_params$learning_rate,
  min_data_in_leaf = best_params$min_data_in_leaf,
  lambda_l2 = best_params$lambda_l2,
  nrounds = best_iter
)
best_tune_params[["LightGBM"]] <- lightgbm_best_params

# 6. Train LightGBM model and time it
start_time_lgb <- Sys.time()
cat(sprintf("Start training LightGBM: %s\n", format(start_time_lgb, "%H:%M:%S")))
lgb_params

# Updated 2025-09-11: add best hyperparameters here
lgb_params$num_leaves <- best_params$num_leaves
lgb_params$learning_rate <- best_params$learning_rate
lgb_params$min_data_in_leaf <- best_params$min_data_in_leaf
lgb_params$lambda_l2 <- best_params$lambda_l2

lgb_model <- lgb.train(
  params = lgb_params,
  data = dtrain_lgb,
  valids = list(test = dtest_lgb),
  nrounds = best_iter,
  early_stopping_rounds = 50,
  eval_freq = 10
)

# Calculate training time
lgb_train_time <- round(as.numeric(difftime(Sys.time(), start_time_lgb, units = "secs")), 1)
cat(sprintf("LightGBM training completed (elapsed: %s sec)\n", lgb_train_time))

# 7. Get predicted probabilities
# Training set predicted probabilities
train_lgb_preds <- predict(lgb_model, train_matrix_lgb)
# Test set predicted probabilities
test_lgb_preds <- predict(lgb_model, test_matrix_lgb)

# 8. Calculate performance metrics and integrate into result lists
algoName <- "LightGBM"

# Training set metrics
train_metrics[[algoName]] <- calculate_metrics(train_labels_lgb, train_lgb_preds)

# Test set metrics
test_metrics[[algoName]] <- calculate_metrics(test_labels_lgb, test_lgb_preds)

# Create ROC objects
roc_train_lgb <- roc(train_labels_lgb, train_lgb_preds)
roc_test_lgb <- roc(test_labels_lgb, test_lgb_preds)

# Add to ROC list
train_roc_list[[algoName]] <- roc_train_lgb
test_roc_list[[algoName]] <- roc_test_lgb

# Add to model container and AUC results
modelContainer[["lightgbm"]] <- lgb_model
AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_lgb$auc)))

# Add to probability list (for DCA)
train_probs_list[[algoName]] <- train_lgb_preds
test_probs_list[[algoName]] <- test_lgb_preds

# === Add LightGBM residual calculation ===
# Training residuals = actual values - predicted probabilities
train_residuals_lgb <- train_labels_lgb - train_lgb_preds
# Test residuals = actual values - predicted probabilities
test_residuals_lgb <- test_labels_lgb - test_lgb_preds

# Store results
residuals_train_list[[algoName]] <- train_residuals_lgb
residuals_test_list[[algoName]] <- test_residuals_lgb

# Record training time
training_times[algoName] <- lgb_train_time

# Performance output
cat(sprintf("  - LightGBM training AUC: %.3f\n", auc(roc_train_lgb)))
cat(sprintf("  - LightGBM test AUC: %.3f\n", auc(roc_test_lgb)))

# Show training time table
cat("\nModel training time summary:\n")
print(data.frame(
  Model = names(training_times),
  Train_time_seconds = unlist(training_times)
))

# Output 1: modeling time statistics
write.csv(data.frame(
  Model = names(training_times),
  Train_time_seconds = unlist(training_times)
),'1.1_ML_training_time_record.csv')

# Create best parameter summary table
best_params_df <- data.frame(
  Model = character(),
  Parameter = character(),
  Value = character(),
  stringsAsFactors = FALSE
)

# Loop through best parameters of all models
for (model_name in names(best_tune_params)) {
  params <- best_tune_params[[model_name]]
  
  if (!is.null(params) && nrow(params) > 0) {
    for (param_name in names(params)) {
      new_row <- data.frame(
        Model = model_name,
        Parameter = param_name,
        Value = as.character(params[[param_name]])
      )
      best_params_df <- rbind(best_params_df, new_row)
    }
  } else {
    new_row <- data.frame(
      Model = model_name,
      Parameter = "No tuning parameters",
      Value = ""
    )
    best_params_df <- rbind(best_params_df, new_row)
  }
}

# Save as CSV file
write.csv(best_params_df, "1.2_Best_Tuning_Parameters.csv", row.names = FALSE)

# If you removed any models earlier, remove their colors here as well.
# Prepare model-color mapping
color_mapping <- c(
  AdaptiveBoosting = "#E74C3C",    
  BayesMethod = "#3498DB",         
  BoostingMethod = "#F39C12",      
  CATBoost = "#27AE60",           
  XGBoost = "#2980B9",            
  DiscriminantModel = "#8E44AD",  
  GradientBoosting = "#16A085",    
  Lasso = "#E67E22",              
  LightGBM = "#A593E0",           
  LogisticModel = "#5D4037",      
  NeighborMethod = "#F1C40F",    
  NeuralNet = "#2C3E50",          
  PLSModel = "#1ABC9C",          
  RandomForest = "#D35400",      
  SVM_Kernel = "#C0392B"          
)

allcolour <- data.frame(color_mapping)$color_mapping

# Output 2:
# === Performance metrics tables ==============================================
library(gridExtra)
library(grid)

# Organize training set metrics into a data frame
train_results <- bind_rows(train_metrics, .id = "Model")

# Rename training set metrics table
colnames(train_results) <- c("Models", 
                             "Sensitivity", "Specificity", "Accuracy", 
                             "PPV", "NPV", 'F1', "Youden's index")

# Save training set performance metrics
write.csv(train_results, "2a_Train_Performance_Metrics.csv", row.names = FALSE)

# Create training set performance table PDF
pdf("2a_Train_Performance_Table.pdf", width = 12, height = 10)
grid.table(train_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()

# === Plot training set metric line plot ===
library(ggplot2)
library(tidyr)

# Convert data to long format
train_long <- gather(train_results, key = "Metric", value = "Value", -Models)

# Create training set metric line plot
train_plot <- ggplot(train_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Training Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        legend.position = "bottom",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = color_mapping)
train_plot 
# Save training set metric line plot
ggsave("3a_Train_Metrics_LinePlot.pdf", train_plot, width = 10, height = 8)
dev.off()

# Organize test set metrics into a data frame
test_results <- bind_rows(test_metrics, .id = "Model")

# Rename test set metrics table
colnames(test_results) <- c("Models", 
                            "Sensitivity", "Specificity", "Accuracy", 
                            "PPV", "NPV", 'F1', "Youden's index")

# Save test set performance metrics
write.csv(test_results, "2b_Test_Performance_Metrics.csv", row.names = FALSE)

# Create test set performance table PDF
pdf("2b_Test_Performance_Table.pdf", width = 12, height = 10)
grid.table(test_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()

test_long <- gather(test_results, key = "Metric", value = "Value", -Models)

# Create test set metric line plot
test_plot <- ggplot(test_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Test Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        legend.position = "bottom",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = color_mapping)
test_plot 
# Save test set metric line plot
ggsave("3b_Test_Metrics_LinePlot.pdf", test_plot, width = 10, height = 8)
dev.off()

# === AUC confidence intervals ==============================================
# Calculate training set AUC confidence intervals
train_auc_ci <- data.frame()
for (model_name in names(train_roc_list)) {
  if (!is.null(train_roc_list[[model_name]])) {
    ci <- ci.auc(train_roc_list[[model_name]])
    train_auc_ci <- rbind(train_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(train_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# Format training set AUC and confidence intervals
train_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                               train_auc_ci$AUC,
                               train_auc_ci$CI_lower,
                               train_auc_ci$CI_upper)

# Save training set AUC confidence interval results
write.csv(train_auc_ci, "4a_Train_AUC_Confidence_Intervals.csv", row.names = FALSE)

# Calculate test set AUC confidence intervals
test_auc_ci <- data.frame()
for (model_name in names(test_roc_list)) {
  if (!is.null(test_roc_list[[model_name]])) {
    ci <- ci.auc(test_roc_list[[model_name]])
    test_auc_ci <- rbind(test_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(test_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# Format test set AUC and confidence intervals
test_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                              test_auc_ci$AUC,
                              test_auc_ci$CI_lower,
                              test_auc_ci$CI_upper)

# Save test set AUC confidence interval results
write.csv(test_auc_ci, "4b_Test_AUC_Confidence_Intervals.csv", row.names = FALSE)

# Output 3: AUC forest plot
# === Forest plot: AUC comparison ======================================
# Prepare data required for forest plot
train_forest_df <- train_auc_ci %>%
  arrange(AUC) %>%
  mutate(
    Model = factor(Model, levels = rev(Model)),
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 

# Create forest plot
ggplot(train_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3, check_overlap = TRUE, show.legend = FALSE) +
  scale_x_continuous(limits = c(min(train_forest_df$CI_lower) * 0.95, max(train_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  labs(title = "Forest Plot of Each Model AUC Score in Trainset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  theme_minimal(base_size = 14) 
# Save forest plot
ggsave("5a_Trainset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")

test_forest_df <- test_auc_ci %>%
  arrange(AUC) %>%
  mutate(
    Model = factor(Model, levels = rev(Model)),
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 

# Create forest plot
ggplot(test_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3, check_overlap = TRUE, show.legend = FALSE) +
  scale_x_continuous(limits = c(min(test_forest_df$CI_lower) * 0.95, max(test_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  labs(title = "Forest Plot of Each Model AUC Score in Testset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  theme_minimal(base_size = 14) 
# Save forest plot
ggsave("5b_Testset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")
dev.off()

# Output 4: ROC curve comparison
# === ROC curve comparison ==============================================
# Training set ROC curve comparison

str(train_roc_list$RandomForest)

pdf("6a_Train_ROC_Comparison.pdf", width = 7, height = 7)
first_index <- which(!sapply(train_roc_list, is.null))[1]
if(is.na(first_index)) {
  plot(0, 0, type="n", xlim=c(0,1), ylim=c(0,1), 
       main="No ROC curves available (Training Set)",
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)")
} else {
  n_curves <- length(train_roc_list)
  curve_colors <- allcolour[1:n_curves]
  
  plot(train_roc_list[[first_index]], col = curve_colors[first_index], lwd=2, legacy.axes = T,
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)", 
       main="Training Set ROC Curves")
  
  for (i in (1:n_curves)) {
    if(i != first_index && !is.null(train_roc_list[[i]])) {
      plot(train_roc_list[[i]], add=TRUE, col=curve_colors[i], lwd=2, legacy.axes = T)
    }
  }
  
  abline(a=0, b=1, lty=3, col="gray")
  
  model_names <- names(train_roc_list)
  auc_text <- sapply(train_roc_list, function(roc) {
    if(!is.null(roc)) sprintf("%.3f", auc(roc)) else "N/A"
  })
  legend_text <- paste0(model_names, ": AUC = ", auc_text)
  
  legend("bottomright", legend=legend_text, 
         col=curve_colors, lwd=2, cex=0.8)
}
dev.off()

# Test set ROC curve comparison
pdf("6b_Test_ROC_Comparison.pdf", width = 7, height = 7)
first_index <- which(!sapply(test_roc_list, is.null))[1]
if(is.na(first_index)) {
  plot(0, 0, type="n", xlim=c(0,1), ylim=c(0,1), 
       main="No ROC curves available (Training Set)",
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)")
} else {
  n_curves <- length(test_roc_list)
  curve_colors <- allcolour[1:n_curves]
  
  plot(test_roc_list[[first_index]], col = curve_colors[first_index], lwd=2, legacy.axes = T,
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)", 
       main="Test Set ROC Curves")
  
  for (i in (1:n_curves)) {
    if(i != first_index && !is.null(test_roc_list[[i]])) {
      plot(test_roc_list[[i]], add=TRUE, col=curve_colors[i], lwd=2, legacy.axes = T)
    }
  }
  
  abline(a=0, b=1, lty=3, col="gray")
  
  model_names <- names(test_roc_list)
  auc_text <- sapply(test_roc_list, function(roc) {
    if(!is.null(roc)) sprintf("%.3f", auc(roc)) else "N/A"
  })
  legend_text <- paste0(model_names, ": AUC = ", auc_text)
  
  legend("bottomright", legend=legend_text, 
         col=curve_colors, lwd=2, cex=0.8)
}
dev.off()

# Output 5: decision curve analysis (DCA)
# === Decision curve analysis (DCA) module =====================================
# Prepare training set data (convert to 0/1 format)
train_truth <- ifelse(train_data$Group == "SZ", 1, 0)

# Create DCA data frame for training set
dca_train_data <- data.frame(truth = train_truth)
for (model_name in names(train_probs_list)) {
  dca_train_data[[model_name]] <- train_probs_list[[model_name]]
}

# Create DCA data frame for test set
dca_test_data <- data.frame(truth = test_group)
for (model_name in names(test_probs_list)) {
  dca_test_data[[model_name]] <- test_probs_list[[model_name]]
}

# === Custom DCA calculation function ===
calculate_dca <- function(data, outcome, predictors) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  results <- data.frame()
  
  outcome_vector <- data[[outcome]]
  n <- length(outcome_vector)
  n_positive <- sum(outcome_vector == 1)
  n_negative <- sum(outcome_vector == 0)
  
  for (pt in thresholds) {
    all_positive_nb <- (n_positive / n) - (n_negative / n) * (pt / (1 - pt))
    all_negative_nb <- 0
    
    model_nbs <- sapply(predictors, function(p) {
      pred <- data[[p]]
      if (all(is.na(pred))) return(NA)
      
      pred_positive <- pred >= pt
      tp <- sum(outcome_vector == 1 & pred_positive, na.rm = TRUE)
      fp <- sum(outcome_vector == 0 & pred_positive, na.rm = TRUE)
      n_valid <- length(which(!is.na(pred)))
      
      (tp / n_valid) - (fp / n_valid) * (pt / (1 - pt))
    })
    
    res_row <- data.frame(
      threshold = pt,
      variable = c("Treat all", "Treat none", names(model_nbs)),
      net_benefit = c(all_positive_nb, all_negative_nb, model_nbs)
    )
    
    results <- rbind(results, res_row)
  }
  
  return(results)
}

# === Use custom function for DCA analysis ===

# Process training set data
dca_train_data_clean <- na.omit(dca_train_data)
model_names_train <- setdiff(colnames(dca_train_data_clean), "truth")
dca_train_res <- calculate_dca(dca_train_data_clean, "truth", model_names_train)

# Process test set data
dca_test_data_clean <- na.omit(dca_test_data)
model_names_test <- setdiff(colnames(dca_test_data_clean), "truth")
dca_test_res <- calculate_dca(dca_test_data_clean, "truth", model_names_test)

# Reconstruct DCA data as wide format
convert_dca_to_wide <- function(dca_data) {
  strategies <- unique(dca_data$variable)
  wide_data <- data.frame(threshold = unique(dca_data$threshold))
  
  for (strategy in strategies) {
    wide_data[[strategy]] <- sapply(wide_data$threshold, function(t) {
      dca_data$net_benefit[dca_data$threshold == t & dca_data$variable == strategy]
    })
  }
  
  return(wide_data)
}

# Convert training and test data
dca_train_wide <- convert_dca_to_wide(dca_train_res)
colnames(dca_train_wide) <- gsub(" ", "_", colnames(dca_train_wide))
dca_test_wide <- convert_dca_to_wide(dca_test_res)
colnames(dca_test_wide) <- gsub(" ", "_", colnames(dca_test_wide))

# Modified DCA plotting function with English-only labels
plot_custom_dca <- function(dca_wide, title) {
  model_names <- setdiff(colnames(dca_wide), c("threshold", "Treat_all", "Treat_none"))
  
  p <- ggplot(dca_wide, aes(x = threshold)) +
    geom_line(aes(y = Treat_all, color = "Treat_all"), linetype = "dashed", linewidth = 0.8) +
    geom_line(aes(y = Treat_none, color = "Treat_none"), linetype = "dashed", linewidth = 0.8)
  
  for (model_name in model_names) {
    p <- p + geom_line(aes(y = .data[[model_name]], color = !!model_name), linewidth = 1)
  }
  
  p <- p +
    scale_color_manual(
      name = "Strategy",
      values = color_mapping,
      breaks = c("Treat_all", "Treat_none", model_names)
    ) +
    labs(
      title = title,
      x = "High risk threshold",
      y = "Net benefit"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.text = element_text(size = 9),
      legend.title = element_text(size = 10, face = "bold")
    ) +
    theme(panel.grid.major = element_line(color = "gray90", linewidth = 0.2),
          panel.grid.minor = element_line(color = "gray95", linewidth = 0.1)) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1), 
                       limits = c(0, 1)) +
    coord_cartesian(ylim = c(-0.1, max(dca_wide[model_names], na.rm = TRUE) * 1.1))
  
  return(p)
}

pdf("7a_DCA_Training_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_train_wide, "Decision Curve Analysis (Training Set)"))
dev.off()

pdf("7b_DCA_Test_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_test_wide, "Decision Curve Analysis (Test Set)"))
dev.off()

# Output 6: reverse cumulative residual plot
# Updated 2025-07-10: added residual visualization
library(DALEX)
library(ggplot2)
library(pROC)

# Define prediction function (outputs probability of positive class)
p_fun <- function(object, newdata) {
  if (inherits(object, "train")) {
    # caret model
    predict(object, newdata = newdata, type = "prob")[, "SZ"]
  } else if (inherits(object, "catboost.Model")) {
    # CatBoost model
    pool <- catboost.load_pool(newdata)
    catboost.predict(object, pool, prediction_type = "Probability")
  } else if (inherits(object, "lgb.Booster")) {
    # LightGBM model
    predict(object, as.matrix(newdata))
  }
}

# Prepare true labels for training and test sets
yTrain <- ifelse(train_data$Group == "SZ", 1, 0)
yTest <- as.numeric(test_group)

# Create lists to store performance objects
train_mp_list <- list()
test_mp_list <- list()

# === Create explainers and compute performance for each model ===

# 1. Process caret-trained models
for (idx in seq_len(nrow(model_settings))) {
  algoName <- model_settings$AlgorithmName[idx]
  model <- modelContainer[[model_settings$Implementation[idx]]]
  
  if (!is.null(model)) {
    # Training set explainer and performance
    explainer_train <- DALEX::explain(
      model = model,
      data = train_data[, -which(names(train_data) == response_var)],
      y = yTrain,
      label = algoName,
      predict_function = p_fun,
      verbose = FALSE
    )
    train_mp_list[[algoName]] <- model_performance(explainer_train)
    # Test set explainer and performance
    explainer_test <- DALEX::explain(
      model = model,
      data = test_data,
      y = yTest,
      label = algoName,
      predict_function = p_fun,
      verbose = FALSE
    )
    test_mp_list[[algoName]] <- model_performance(explainer_test)
  }
}

# 2. Process CatBoost model
if (!is.null(modelContainer[["catboost"]])) {
  algoName <- "CATBoost"
  model <- modelContainer[["catboost"]]
  
  # Training set explainer and performance
  explainer_train <- DALEX::explain(
    model = model,
    data = train_data[, -which(names(train_data) == response_var)],
    y = yTrain,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  train_mp_list[[algoName]] <- model_performance(explainer_train)
  
  # Test set explainer and performance
  explainer_test <- DALEX::explain(
    model = model,
    data = test_data,
    y = yTest,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  test_mp_list[[algoName]] <- model_performance(explainer_test)
}

# 3. Process LightGBM model
if (!is.null(modelContainer[["lightgbm"]])) {
  algoName <- "LightGBM"
  model <- modelContainer[["lightgbm"]]
  
  # Training set explainer and performance
  explainer_train <- DALEX::explain(
    model = model,
    data = train_data[, -which(names(train_data) == response_var)],
    y = yTrain,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  train_mp_list[[algoName]] <- model_performance(explainer_train)
  
  # Test set explainer and performance
  explainer_test <- DALEX::explain(
    model = model,
    data = test_data,
    y = yTest,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  test_mp_list[[algoName]] <- model_performance(explainer_test)
}

plot(train_mp_list[[1]],train_mp_list[[2]],train_mp_list[[3]],
     train_mp_list[[4]],train_mp_list[[5]],train_mp_list[[6]],
     train_mp_list[[7]],train_mp_list[[8]],train_mp_list[[9]],
     train_mp_list[[10]],train_mp_list[[11]],train_mp_list[[12]],
     train_mp_list[[13]],train_mp_list[[14]],train_mp_list[[15]])+
  scale_color_manual(values = color_mapping) + ggtitle('Reverse_cumulative_|residua|_distribution_trainset')
ggsave('8a_Reverse_cumulative_residual_distribution_train_set.pdf',width = 9.7, height = 6.1)

plot(test_mp_list[[1]],test_mp_list[[2]],test_mp_list[[3]],
     test_mp_list[[4]],test_mp_list[[5]],test_mp_list[[6]],
     test_mp_list[[7]],test_mp_list[[8]],test_mp_list[[9]],
     test_mp_list[[10]],test_mp_list[[11]],test_mp_list[[12]],
     test_mp_list[[13]],test_mp_list[[14]],train_mp_list[[15]])+
  scale_color_manual(values = color_mapping)+ ggtitle('Reverse_cumulative_|residua|_distribution_testset')
ggsave('8b_Reverse_cumulative_residual_distribution_test_set.pdf',width = 9.7, height = 6.1)

# Output 7: residual boxplot
plot(train_mp_list[[1]],train_mp_list[[2]],train_mp_list[[3]],
     train_mp_list[[4]],train_mp_list[[5]],train_mp_list[[6]],
     train_mp_list[[7]],train_mp_list[[8]],train_mp_list[[9]],
     train_mp_list[[10]],train_mp_list[[11]],train_mp_list[[12]],
     train_mp_list[[13]],train_mp_list[[14]],train_mp_list[[15]],geom = "boxplot")+
  scale_fill_manual(values = color_mapping)+ggtitle('Boxplot of |Residual| in traindata')
ggsave('9a_Boxplot_redisual_train_set.pdf',width = 9.42, height = 6.61)

plot(test_mp_list[[1]],test_mp_list[[2]],test_mp_list[[3]],
     test_mp_list[[4]],test_mp_list[[5]],test_mp_list[[6]],
     test_mp_list[[7]],test_mp_list[[8]],test_mp_list[[9]],
     test_mp_list[[10]],test_mp_list[[11]],test_mp_list[[12]],
     test_mp_list[[13]],test_mp_list[[14]],train_mp_list[[15]],geom = "boxplot")+
  scale_fill_manual(values = color_mapping)+ggtitle('Boxplot of |Residual| in testdata')
ggsave('9b_Boxplot_redisual_test_set.pdf',width = 9.42, height = 6.61)

AUCresults
# === Find the best model ===
best_model_name <- test_auc_ci$Model[which.max(test_auc_ci$AUC)]
cat(sprintf("Best model: %s (AUC = %.3f)\n", best_model_name, max(test_auc_ci$AUC)))

## Updated 2025-07-09: support SHAP interpretation for best models CATBoost and LightGBM
# best_model_name = "CATBoost"
best_model_name = "CATBoost"

# Get best model object and prepare prediction functions
if (best_model_name == "CATBoost") {
  final_model <- modelContainer[["catboost"]]
  
  # CatBoost-specific prediction function
  predict_catboost <- function(model, newdata) {
    pool <- catboost.load_pool(newdata)
    probs <- catboost.predict(model, pool, prediction_type = "Probability")
    pred_labels <- ifelse(probs > 0.5, "SZ", "HC")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("HC", "SZ"))))
  }
  
} else if (best_model_name == "LightGBM") {
  final_model <- modelContainer[["lightgbm"]]
  
  # LightGBM-specific prediction function
  predict_lightgbm <- function(model, newdata) {
    probs <- predict(model, as.matrix(newdata))
    pred_labels <- ifelse(probs > 0.5, "SZ", "HC")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("HC", "SZ"))))
  }
  
} else {
  impl <- model_settings$Implementation[which(model_settings$AlgorithmName == best_model_name)]
  final_model <- train(formula, train_data,
                       method = impl,
                       trControl = cv_control)
  
  # caret-specific prediction function
  predict_caret <- function(model, newdata) {
    probs <- predict(model, newdata = newdata, type = "prob")[,"SZ"]
    pred_labels <- predict(model, newdata = newdata)
    return(list(probs = probs, labels = pred_labels))
  }
}

# Output 8: best model confusion matrix
# === Best model confusion matrix visualization ===
# 1. Training set confusion matrix
cat("\n===== Plot training set confusion matrix =====\n")
if (best_model_name == "CATBoost") {
  pred_result <- predict_catboost(final_model, train_data[, -which(names(train_data) == response_var)])
} else if (best_model_name == "LightGBM") {
  pred_result <- predict_lightgbm(final_model, train_data[, -which(names(train_data) == response_var)])
} else {
  pred_result <- predict_caret(final_model, train_data)
}

train_pred <- pred_result$labels
train_probs <- pred_result$probs
train_truth <- train_data$Group

# Ensure factor levels are consistent
train_truth <- factor(ifelse(train_truth == "HC", "HC", "SZ"),
                      levels = c("HC", "SZ"))

# Create confusion matrix
train_cm <- confusionMatrix(train_pred, train_truth, positive = "SZ")
# Calculate accuracy
accuracy <- round(train_cm$overall["Accuracy"] * 100, 1)

train_cm_data <- as.data.frame(train_cm$table)
train_cm_data$AccuracyLabel <- ifelse(train_cm_data$Reference == train_cm_data$Prediction, "Correct", "Incorrect")

# Improved training set confusion matrix
ggplot(train_cm_data, aes(x = Prediction, y = Reference, fill = AccuracyLabel)) +
  geom_tile(color = "white", alpha = 0.9, width = 0.95, height = 0.95) +
  geom_text(aes(label = Freq, color = AccuracyLabel), 
            size = 8, fontface = "bold", show.legend = FALSE) +
  scale_fill_manual(values = c(Correct = "#4da6ff", Incorrect = "#E66767")) +
  scale_color_manual(values = c(Correct = "white", Incorrect = "black")) +
  labs(title = "Training Set Confusion Matrix",
       subtitle = sprintf("Accuracy: %.1f%%", accuracy),
       x = "Predicted Label",
       y = "True Label",
       fill = "") +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "none",
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#e0e0e0", fill = NA, size = 1.5),
    aspect.ratio = 1
  ) 

# Save training set confusion matrix
ggsave("10a_Best_Model_Confusion_Matrix_Training_Set.pdf",
       width = 10, height = 8, device = "pdf")

# 2. Test set confusion matrix
cat("\n===== Plot test set confusion matrix =====\n")
if (best_model_name == "CATBoost") {
  pred_result <- predict_catboost(final_model, test_data)
} else if (best_model_name == "LightGBM") {
  pred_result <- predict_lightgbm(final_model, test_data)
} else {
  pred_result <- predict_caret(final_model, test_data)
}

test_pred <- pred_result$labels
test_probs <- pred_result$probs
test_truth <- factor(ifelse(test_group == "1", "SZ", "HC"),
                     levels = c("HC", "SZ"))

# Create confusion matrix
test_cm <- confusionMatrix(test_pred, test_truth, positive = "SZ")

# Calculate accuracy
accuracy_test <- round(test_cm$overall["Accuracy"] * 100, 1)

test_cm_data <- as.data.frame(test_cm$table)
test_cm_data$AccuracyLabel <- ifelse(test_cm_data$Reference == test_cm_data$Prediction, "Correct", "Incorrect")

# Improved test set confusion matrix
ggplot(test_cm_data, aes(x = Prediction, y = Reference, fill = AccuracyLabel)) +
  geom_tile(color = "white", alpha = 0.9, width = 0.95, height = 0.95) +
  geom_text(aes(label = Freq, color = AccuracyLabel), 
            size = 8, fontface = "bold", show.legend = FALSE) +
  scale_fill_manual(values = c(Correct = "#4da6ff", Incorrect = "#E66767")) +
  scale_color_manual(values = c(Correct = "white", Incorrect = "black")) +
  labs(title = "Test Set Confusion Matrix",
       subtitle = sprintf("Accuracy: %.1f%%", accuracy_test),
       x = "Predicted Label",
       y = "True Label",
       fill = "") +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "none",
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#e0e0e0", fill = NA, size = 1.5),
    aspect.ratio = 1
  ) 

# Save test set confusion matrix
ggsave("10b_Best_Model_Confusion_Matrix_Test_Set.pdf",
       width = 10, height = 8, device = "pdf")
dev.off()

# === Clinical impact curve and calibration curve module ===

# Output 9: clinical impact curve
# Load required packages
library(ggplot2)
library(reshape2)

# Clinical impact curve function
calculate_clinical_impact <- function(probs_list, truth) {
  thresholds <- seq(0, 1, by = 0.01)
  impact_data <- data.frame(Threshold = thresholds)
  
  for (model_name in names(probs_list)) {
    pred_probs <- probs_list[[model_name]]
    if (all(is.na(pred_probs))) next
    
    pred_positives <- sapply(thresholds, function(t) sum(pred_probs >= t, na.rm = TRUE))
    
    true_positives <- sapply(thresholds, function(t) {
      idx <- which(pred_probs >= t)
      sum(truth[idx] == 1, na.rm = TRUE)
    })
    
    impact_data[[model_name]] <- pred_positives
    impact_data[[paste0(model_name, "_Observed")]] <- true_positives
  }
  
  return(impact_data)
}

# Calibration curve calculation function
calculate_calibration_curve <- function(probs_list, truth, n_bins = 10) {
  calibration_data <- data.frame()
  
  for (model_name in names(probs_list)) {
    pred_probs <- probs_list[[model_name]]
    if (all(is.na(pred_probs))) next
    
    valid_idx <- !is.na(pred_probs)
    pred_probs <- pred_probs[valid_idx]
    truth_sub <- truth[valid_idx]
    
    quantiles <- quantile(pred_probs, probs = seq(0, 1, length.out = n_bins + 1))
    bins <- cut(pred_probs, breaks = quantiles, include.lowest = TRUE)
    
    obs_rate <- tapply(truth_sub, bins, mean)
    pred_mean <- tapply(pred_probs, bins, mean)
    bin_count <- tapply(pred_probs, bins, length)
    
    calibration_data <- rbind(calibration_data, 
                              data.frame(
                                Model = model_name,
                                Predicted = pred_mean,
                                Observed = obs_rate,
                                Count = bin_count
                              ))
  }
  
  return(calibration_data)
}

# Create probability lists for the best model
best_train_probs_list <- list()
best_train_probs_list[[best_model_name]] <- train_probs_list[[best_model_name]]
best_test_probs_list <- list()
best_test_probs_list[[best_model_name]] <- test_probs_list[[best_model_name]]

# === Use the same parameters in the calling section ===
train_truth <- ifelse(train_data$Group == "SZ", 1, 0)
test_truth <- as.numeric(test_group)

# Calculate clinical impact curves using new function
best_train_impact_data <- calculate_clinical_impact(best_train_probs_list, train_truth)
best_test_impact_data <- calculate_clinical_impact(best_test_probs_list, test_truth)

head(best_train_impact_data)
head(best_test_impact_data)

# Calculate calibration curves using new function
train_calibration_data <- calculate_calibration_curve(best_train_probs_list, train_truth, n_bins = 10)
test_calibration_data <- calculate_calibration_curve(best_test_probs_list, test_truth, n_bins = 10)
head(train_calibration_data)
head(test_calibration_data)

# 1. Clinical impact curve visualization function
plot_clinical_impact <- function(impact_data, title) {
  long_data <- melt(impact_data, id.vars = "Threshold", variable.name = "Variable", value.name = "Count")
  
  long_data$Type <- ifelse(grepl("_Observed", long_data$Variable), "Observed", "Predicted")
  long_data$Model <- gsub("_Observed", "", long_data$Variable)
  
  model_name <- unique(long_data$Model)
  
  p <- ggplot(long_data, aes(x = Threshold, y = Count, color = Type, linetype = Type)) +
    geom_line(size = 1.2) +
    scale_color_manual(values = c("Predicted" = "#E41A1C", "Observed" = "#377EB8")) +
    scale_linetype_manual(values = c("Predicted" = "solid", "Observed" = "dashed")) +
    labs(title = paste(title, "-", model_name),
         x = "High Risk Threshold", 
         y = "Number of Cases",
         color = "",
         linetype = "") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 12),
      legend.position = "top",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 1)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.2))
  
  return(p)
}

# 2. Calibration curve visualization function
plot_calibration_curve <- function(calibration_data, title) {
  calibration_data$PointSize <- sqrt(calibration_data$Count) * 0.8
  
  p <- ggplot(calibration_data, aes(x = Predicted, y = Observed)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40", size = 1) +
    geom_point(aes(size = PointSize, color = Model), alpha = 0.8) +
    geom_line(aes(color = Model), size = 1.0, alpha = 0.7) +
    scale_color_manual(values = color_mapping) +
    scale_size(range = c(3, 8), guide = "none") +
    labs(title = paste("Calibration Curve:", title),
         x = "Mean Predicted Probability", 
         y = "Observed Event Rate",
         color = "Model") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 12),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 1)
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    annotate("text", x = 0.8, y = 0.2, label = "Perfect calibration", 
             color = "gray40", size = 5)
  
  return(p)
}

# 3. Create and save all plots
# Training set clinical impact curve
train_impact_plot <- plot_clinical_impact(
  best_train_impact_data, "Training Set"
)
train_impact_plot
ggsave("11a_Clinical_Impact_Training_Set_Best_Model.pdf", 
       train_impact_plot, width = 9, height = 7)

# Test set clinical impact curve
test_impact_plot <- plot_clinical_impact(
  best_test_impact_data, "Test Set"
)
test_impact_plot 
ggsave("11b_Clinical_Impact_Test_Set_Best_Model.pdf", 
       test_impact_plot, width = 9, height = 7)

# Output 10: calibration curve
# Training set calibration curve
train_calibration_plot <- plot_calibration_curve(
  train_calibration_data, "Training Set"
)

train_calibration_plot 
ggsave("12a_Calibration_Curve_Training_Set_Best_Model.pdf", 
       train_calibration_plot, width = 8, height = 7)

# Test set calibration curve
test_calibration_plot <- plot_calibration_curve(
  test_calibration_data, "Test Set"
)
test_calibration_plot 
ggsave("12b_Calibration_Curve_Test_Set_Best_Model.pdf", 
       test_calibration_plot, width = 8, height = 7)
dev.off()

# Final output: SHAP
# SHAP visualization
# Create custom prediction function (outputs numeric probabilities)
# Create unified prediction function
# === SHAP visualization section ===
# Create unified prediction function
pred_wrapper <- function(model, model_type, newdata) {
  if (model_type == "caret") {
    return(predict(model, newdata = newdata, type = "prob")[,"SZ"])
  } else if (model_type == "catboost") {
    pool <- catboost.load_pool(newdata)
    return(catboost.predict(model, pool, prediction_type = "Probability"))
  } else if (model_type == "lightgbm") {
    return(predict(model, as.matrix(newdata)))
  }
}

# Identify best model type
best_model_type <- case_when(
  best_model_name %in% model_settings$AlgorithmName ~ "caret",
  best_model_name == "CATBoost" ~ "catboost",
  best_model_name == "LightGBM" ~ "lightgbm"
)

# Calculate SHAP values
shap_vis <- NULL

if (best_model_type == "caret") {
  # SHAP calculation for caret models
  shap_values <- kernelshap(
    final_model, 
    X = train_data[, -which(names(train_data) == response_var)],
    pred_fun = function(m, x) pred_wrapper(m, "caret", x)
  )
  shap_vis <- shapviz(shap_values, train_data[, -which(names(train_data) == response_var)])
  
} else if (best_model_type == "catboost") {
  # CatBoost-specific SHAP calculation
  pool_train <- catboost.load_pool(train_data[, -which(names(train_data) == response_var)])
  shap_values_matrix <- catboost.get_feature_importance(
    final_model, 
    pool_train, 
    type = "ShapValues"
  )
  # Extract SHAP values and baseline
  shap_values <- shap_values_matrix[, -ncol(shap_values_matrix)]
  baseline <- shap_values_matrix[1, ncol(shap_values_matrix)]
  colnames(shap_values) <- colnames(train_data[, -which(names(train_data) == response_var)])
  
  shap_vis <- shapviz(
    shap_values, 
    X = train_data[, -which(names(train_data) == response_var)],
    baseline = baseline)
  cat("CATBoost SHAP calculation completed\n")
  
} else if (best_model_type == "lightgbm") {
  # Alternative SHAP calculation for LightGBM
  if (!requireNamespace("fastshap", quietly = TRUE)) {
    install.packages("fastshap")
  }
  library(fastshap)
  
  # Define LightGBM prediction function
  pred_fun_lightgbm <- function(object, newdata) {
    predict(object, as.matrix(newdata))
  }
  
  # Calculate SHAP values
  shap_values <- fastshap::explain(
    final_model, 
    X = as.data.frame(train_data[, -which(names(train_data) == response_var)]),
    pred_wrapper = pred_fun_lightgbm,
    nsim = 10
  )
  
  # Create shapviz object
  shap_vis <- shapviz(shap_values, X = train_data[, -which(names(train_data) == response_var)])
  cat("LightGBM SHAP calculation completed\n")
}

# Rank feature importance
feature_importance <- colMeans(abs(shap_vis$S))
sorted_features <- names(sort(feature_importance, decreasing = TRUE))

# Visualization settings
visualization_theme <- theme_minimal() + 
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12))

# Generate model interpretation plots - SHAP visualizations (English)

# 1. Feature importance bar plot (global explanation)
#    Shows the average absolute impact of each feature on model output.
#    Larger values indicate greater feature importance.
#    Numeric labels allow quantitative comparison.
pdf("13_SHAP_Feature_Importance_Barplot.pdf", width=8, height=6)
sv_importance(shap_vis, kind="bar", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "Feature Importance (Mean Absolute SHAP)",
       subtitle = "Average impact magnitude of each feature on model predictions",
       x = "Mean |SHAP value|", y = "Feature",
       caption = "Bar height indicates feature importance, with value showing mean absolute SHAP")
dev.off()

# 2. Bee swarm plot (feature effect distribution)
#    Shows the distribution of SHAP values for each feature across the dataset.
#    Color indicates feature value (red high, blue low).
#    Displays the relationship trend between feature values and predictions.
#    More spread indicates stronger nonlinear relationships.
pdf("14_SHAP_BeeSwarm_Plot.pdf", width=9, height=7)
sv_importance(shap_vis, kind="bee", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "SHAP Value Distribution (Bee Swarm)",
       subtitle = "Each point represents one sample, color indicates feature value",
       x = "SHAP value (impact on model output)",
       y = "Feature",
       caption = "Red: high feature values | Blue: low feature values | Horizontal spread: direction of effect")
dev.off()

# 3. Feature dependence plots (multiple features)
#    Shows nonlinear relationships between feature values and SHAP values.
#    Each subplot displays the dependence of one feature on model output.
#    Trend lines help interpret relationship patterns.
pdf("15_SHAP_Feature_Dependence.pdf", width=10, height=8)
sv_dependence(shap_vis, sorted_features[1:9]) +
  visualization_theme
dev.off()

# 4. Waterfall plot (single-sample explanation)
dir.create('16_Sample_Waterfall', showWarnings = FALSE)
for (i in 1:min(10, nrow(train_data))) {
  pdf(paste0("16_Sample_Waterfall/sample_", i, "_waterfall.pdf"), width=9, height=6)
  p <- sv_waterfall(shap_vis, row_id = i) +  
    labs(title = paste("Prediction Breakdown for Sample", i),
         subtitle = "Cumulative contribution of features to final prediction")
  print(p)
  dev.off()
}

# 5. Force plot for individual samples
dir.create('17_Force_Plot', showWarnings = FALSE)
for (i in 1:min(10, nrow(train_data))) {
  pdf(paste0("17_Force_Plot/sample_", i, "_force.pdf"), width=9, height=6)
  p <- sv_force(shap_vis, row_id = i) +  
    labs(title = paste("Feature Contributions for Sample", i),
         subtitle = "Visual forces pushing prediction from base value to output")
  print(p)
  dev.off()
}