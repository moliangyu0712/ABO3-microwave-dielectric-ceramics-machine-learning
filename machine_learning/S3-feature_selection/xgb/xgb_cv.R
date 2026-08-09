library(Metrics)
library(xgboost)
library(Matrix)
library(foreach)
library(doParallel)

get_env <- function(name, default) {
  val <- Sys.getenv(name)
  if (val == "") return(default)
  return(val)
}

project_root       <- get_env("PROJECT_ROOT", "D:/MLY/ABO3/code/machine_learning")
s3_output_dir      <- get_env("S3_OUTPUT_DIR", file.path(project_root, "output/S3-feature_selection"))

dire               <- get_env("FEATURE_COMB_DIR", file.path(s3_output_dir, "feature_combination"))
feature_file       <- get_env("FEATURE_COMB_FILE", file.path(s3_output_dir, "feature_combination.csv"))
xgb_hp_output_dir  <- get_env("XGB_HP_OUTPUT_DIR", file.path(s3_output_dir, "xgb/hp"))

target_col         <- get_env("TARGET", "targ")
train_ratio        <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
core               <- as.numeric(get_env("XGB_N_CORES", "10"))
cv_folds           <- as.numeric(get_env("XGB_CV_FOLDS", "10"))
cv_repeats         <- as.numeric(get_env("XGB_CV_REPEATS", "50"))
nrounds            <- as.numeric(get_env("XGB_NROUND_OPTIMAL", "25"))

max_depth_str      <- get_env("XGB_MAX_DEPTH_VALUES", "2,4,6,8,10")
learning_rate_str  <- get_env("XGB_LEARNING_RATE_VALUES", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")

max_depth_list     <- as.numeric(strsplit(max_depth_str, ",")[[1]])
learning_rate_list <- as.numeric(strsplit(learning_rate_str, ",")[[1]])

if (!endsWith(xgb_hp_output_dir, "/")) {
  xgb_hp_output_dir <- paste0(xgb_hp_output_dir, "/")
}

cl <- makeCluster(core)
registerDoParallel(cl)

if (!file.exists(feature_file)) {
  stop(paste("Feature combination file not found: ", feature_file))
}
feature <- read.csv(feature_file, encoding = "UTF-8")
names(feature)[1] <- "feature"
L <- nrow(feature)

prob_val <- c(train_ratio, 1 - train_ratio)
nrow_val <- length(max_depth_list)
ncol_val <- length(learning_rate_list)

foreach(tt = 1:cv_repeats, .packages = c("Metrics", "xgboost", "Matrix"), .inorder = FALSE) %dopar% {
  
  for (b in 1:L) {
    name <- feature[b, 1]
    
    file_check1 <- paste(xgb_hp_output_dir, tt, name, "rmse_train.csv")
    file_check2 <- paste(xgb_hp_output_dir, tt, name, "r2_train.csv")
    file_check3 <- paste(xgb_hp_output_dir, tt, name, "rmse_test.csv")
    file_check4 <- paste(xgb_hp_output_dir, tt, name, "r2_test.csv")
    
    if (file.exists(file_check1) && file.exists(file_check2) && 
        file.exists(file_check3) && file.exists(file_check4)) {
      next
    }
    
    data_path <- file.path(dire, paste0(name, ".csv"))
    if (!file.exists(data_path)) {
      next
    }
    data <- read.csv(data_path, encoding = "UTF-8")
    
    aver1_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver2_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver1_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver2_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    
    set.seed(tt)
    par1 <- sample(2, nrow(data), replace = TRUE, prob = prob_val)
    train1 <- data[par1 == 1, ]
    
    folds <- cut(seq_len(nrow(train1)), breaks = cv_folds, labels = FALSE)
    
    nn <- 1
    for (max_depth in max_depth_list) {
      mm <- 1
      for (learning_rate in learning_rate_list) {
        
        rmse_train <- numeric(cv_folds)
        r2_train <- numeric(cv_folds)
        rmse_test <- numeric(cv_folds)
        r2_test <- numeric(cv_folds)
        
        for (t in 1:cv_folds) {
          train <- train1[folds != t, ]
          test  <- train1[folds == t, ]
          
          formula_str <- paste(target_col, "~ . - 1")
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
          
          set.seed(tt)
          model <- xgb.train(params, dtrain, nrounds = nrounds)
          
          pred_train <- predict(model, dtrain)
          pred_test  <- predict(model, dtest)
          
          rmse_train[t] <- rmse(train[[target_col]], pred_train)
          rmse_test[t]  <- rmse(test[[target_col]], pred_test)
          r2_train[t]   <- summary(lm(train[[target_col]] ~ pred_train))$r.squared
          r2_test[t]    <- summary(lm(test[[target_col]] ~ pred_test))$r.squared
        }
        
        aver1_tr[nn, mm] <- mean(rmse_train)
        aver2_tr[nn, mm] <- mean(r2_train)
        aver1_te[nn, mm] <- mean(rmse_test)
        aver2_te[nn, mm] <- mean(r2_test)
        
        mm <- mm + 1
      }
      nn <- nn + 1
    }
    
    write.csv(aver1_tr, paste(xgb_hp_output_dir, tt, name, "rmse_train.csv"), row.names = FALSE)
    write.csv(aver2_tr, paste(xgb_hp_output_dir, tt, name, "r2_train.csv"), row.names = FALSE)
    write.csv(aver1_te, paste(xgb_hp_output_dir, tt, name, "rmse_test.csv"), row.names = FALSE)
    write.csv(aver2_te, paste(xgb_hp_output_dir, tt, name, "r2_test.csv"), row.names = FALSE)
  }
}

stopCluster(cl)