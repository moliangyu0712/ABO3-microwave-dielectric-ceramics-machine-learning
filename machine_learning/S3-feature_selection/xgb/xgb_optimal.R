library(xgboost)
library(Matrix)
library(Metrics)
library(data.table)
library(foreach)
library(doParallel)
library(dplyr)

r2_machine_learning <- function(y_true, y_pred) {
  sse <- sum((y_true - y_pred)^2)
  sst <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - sse / sst
  return(r2)
}

get_env <- function(name, default) {
  val <- Sys.getenv(name)
  if (val == "") return(default)
  return(val)
}

project_root       <- get_env("PROJECT_ROOT", "D:/MLY/ABO3/code/machine_learning")
s3_output_dir      <- get_env("S3_OUTPUT_DIR", file.path(project_root, "output/S3-feature_selection"))

dire               <- get_env("FEATURE_COMB_DIR", file.path(s3_output_dir, "feature_combination"))
feature_file       <- get_env("FEATURE_COMB_FILE", file.path(s3_output_dir, "feature_combination.csv"))
para_file          <- get_env("XGB_PARA_FILE", file.path(s3_output_dir, "xgb/r2_te.csv"))
xgb_output_dir     <- get_env("XGB_OUTPUT_DIR", file.path(s3_output_dir, "xgb"))

target_col         <- get_env("TARGET", "targ")
train_ratio        <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
core               <- as.numeric(get_env("XGB_N_CORES", "10"))
optimal_repeats    <- as.numeric(get_env("XGB_OPTIMAL_REPEATS", "1000"))
nrounds            <- as.numeric(get_env("XGB_NROUND_OPTIMAL", "25"))

if (!file.exists(feature_file)) {
  stop(paste("Feature combination file not found: ", feature_file))
}
feature <- read.csv(feature_file, encoding = "UTF-8")
names(feature)[1] <- "feature"
L <- dim(feature)[1]

if (!file.exists(para_file)) {
  stop(paste("Hyperparameter config file not found: ", para_file))
}
para <- read.csv(para_file, encoding = "UTF-8")

cl <- makeCluster(core)
registerDoParallel(cl)

prob_val <- c(train_ratio, 1 - train_ratio)

final_result <- foreach(
  b = 1:L,
  .packages = c("xgboost", "Matrix", "Metrics", "dplyr"),
  .combine = rbind
) %dopar% {
  
  max_depth = para[b, 3]
  learning_rate = para[b, 4]
  
  name <- (feature[b, 1])
  file_name = paste(name, "csv", sep = ".")
  data_path = file.path(dire, file_name)
  
  if (!file.exists(data_path)) {
    return(NULL)
  }
  data <- read.csv(data_path, encoding = 'UTF-8')
  
  data$sample_id <- 1:nrow(data)
  
  train_pred <- data.frame(sample_id = data$sample_id)
  test_pred  <- data.frame(sample_id = data$sample_id)
  
  for (a in 1:optimal_repeats) {
    set.seed(a)
    par <- sample(2, nrow(data), replace = TRUE, prob = prob_val)
    train <- data[par == 1, ]
    test  <- data[par == 2, ]
    
    formula_str <- paste(target_col, "~ . - sample_id - 1")
    formula_obj <- as.formula(formula_str)
    
    train_matrix <- sparse.model.matrix(formula_obj, data = train)
    test_matrix  <- sparse.model.matrix(formula_obj, data = test)
    
    dtrain <- xgb.DMatrix(train_matrix, label = train[[target_col]])
    dtest  <- xgb.DMatrix(test_matrix, label = test[[target_col]])
    
    params <- list(
      max_depth = max_depth,
      learning_rate = learning_rate,
      objective = "reg:squarederror",
      nthread = 1,
      verbosity = 0
    )
    
    set.seed(a)
    xgb_model <- xgb.train(params, dtrain, nrounds = nrounds)
    
    ptrain <- predict(xgb_model, dtrain)
    ptest  <- predict(xgb_model, dtest)
    
    train_df <- data.frame(sample_id = train$sample_id, pred = ptrain)
    test_df  <- data.frame(sample_id = test$sample_id,  pred = ptest)
    
    train_pred <- left_join(train_pred, train_df, by = "sample_id")
    test_pred  <- left_join(test_pred, test_df,   by = "sample_id")
    colnames(train_pred)[ncol(train_pred)] <- paste0("pred_", a)
    colnames(test_pred)[ncol(test_pred)]   <- paste0("pred_", a)
  }
  
  train_final <- train_pred %>%
    rowwise() %>%
    mutate(avg_pred = mean(c_across(-sample_id), na.rm = TRUE),
           true_Qf = data[[target_col]][match(sample_id, data$sample_id)]) %>%
    filter(!is.na(avg_pred))
  
  test_final <- test_pred %>%
    rowwise() %>%
    mutate(avg_pred = mean(c_across(-sample_id), na.rm = TRUE),
           true_Qf = data[[target_col]][match(sample_id, data$sample_id)]) %>%
    filter(!is.na(avg_pred))
  
  tr_rmse <- rmse(train_final$true_Qf, train_final$avg_pred)
  tr_r2   <- r2_machine_learning(train_final$true_Qf, train_final$avg_pred)
  tr_mae  <- mae(train_final$true_Qf, train_final$avg_pred)
  
  te_rmse <- rmse(test_final$true_Qf, test_final$avg_pred)
  te_r2   <- r2_machine_learning(test_final$true_Qf, test_final$avg_pred)
  te_mae  <- mae(test_final$true_Qf, test_final$avg_pred)
  
  data.frame(
    feature_no = b,
    formula = name,
    max_depth = max_depth,
    eta = learning_rate,
    train_rmse = tr_rmse,
    train_r2   = tr_r2,
    train_mae  = tr_mae,
    test_rmse  = te_rmse,
    test_r2    = te_r2,
    test_mae   = te_mae
  )
}

stopCluster(cl)

out_file <- file.path(xgb_output_dir, "final_sample_average_metrics.csv")
write.csv(final_result, out_file, row.names = FALSE)
print(paste("✅ Monte Carlo best metric evaluation saved to :", out_file))