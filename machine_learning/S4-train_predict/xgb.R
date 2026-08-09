library(xgboost)
library(Matrix)
library(Metrics)

get_env <- function(name, default) {
  val <- Sys.getenv(name)
  if (val == "") return(default)
  return(val)
}

input_csv    <- get_env("S4_INPUT_CSV", "")
output_dir   <- get_env("S4_MODEL_DIR", "")
formula_str  <- get_env("S4_FORMULA", "")
target_col   <- get_env("TARGET", "targ")
train_ratio  <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
n_repeats    <- as.numeric(get_env("S4_REPEATS", "1000"))

max_depth     <- as.numeric(get_env("S4_HP_P1", "10"))
learning_rate <- as.numeric(get_env("S4_HP_P2", "0.1"))
nrounds_val   <- as.numeric(get_env("XGB_NROUND_OPTIMAL", "25"))

if (input_csv == "" || output_dir == "" || formula_str == "") {
  stop("Missing environment variables for S4 XGB simulation.")
}

data <- read.csv(input_csv, encoding = "UTF-8")
L <- dim(data)[1]

if (!endsWith(output_dir, "/") && !endsWith(output_dir, "\\")) {
  output_dir <- paste0(output_dir, "/")
}

rmse_train <- matrix(0, nrow = 1, ncol = n_repeats)
r2_train   <- matrix(0, nrow = 1, ncol = n_repeats)
rmse_test  <- matrix(0, nrow = 1, ncol = n_repeats)
r2_test    <- matrix(0, nrow = 1, ncol = n_repeats)
pred       <- matrix(0, nrow = L, ncol = n_repeats)
pred2      <- matrix(0, nrow = L, ncol = n_repeats)

formula_matrix <- as.formula(paste(target_col, "~ . - 1"))
prob_val <- c(train_ratio, 1 - train_ratio)

for (a in 1:n_repeats){
  set.seed(a)
  par <- sample(2, nrow(data), replace = TRUE, prob = prob_val)
  train <- data[par == 1, ]
  test  <- data[par == 2, ]
  
  train_matrix <- sparse.model.matrix(formula_matrix, data = train)
  test_matrix  <- sparse.model.matrix(formula_matrix, data = test)
  train_label  <- train[[target_col]]
  test_label   <- test[[target_col]]
  
  dtrain       <- xgb.DMatrix(data = train_matrix, label = train_label)
  dtest        <- xgb.DMatrix(data = test_matrix, label = test_label)
  
  set.seed(a)
  params <- list(
            max_depth = max_depth,
            learning_rate = learning_rate,
            objective = "reg:squarederror",
            nthread = 1,
            verbosity = 0
          )
          
  set.seed(a)
  xgb_model <- xgb.train(params, dtrain, nrounds = nrounds_val)
  
  ptrain <- predict(xgb_model, dtrain)
  rmse_train[1, a] <- rmse(train[[target_col]], ptrain)
  R2a <- data.frame(ptrain = ptrain, targ_val = train[[target_col]])
  la <- lm(targ_val ~ ptrain, data = R2a)
  r2_train[1, a] <- as.numeric(summary(la)$r.squared)
  
  ptest <- predict(xgb_model, dtest)
  rmse_test[1, a] <- rmse(test[[target_col]], ptest)
  R2b <- data.frame(ptest = ptest, targ_val = test[[target_col]])
  lb <- lm(targ_val ~ ptest, data = R2b)
  r2_test[1, a] <- as.numeric(summary(lb)$r.squared)
  
  pp_test <- data.frame(ptest, row.names = row.names(test))
  pp_test <- as.matrix(pp_test)
  for (k in 1:L){
    if (k %in% row.names(pp_test)) {
      pred[k, a] = pp_test[row.names(pp_test) == k]
    } else {
      pred[k, a] = NA
    }
  }
  
  pp_train <- data.frame(ptrain, row.names = row.names(train))
  pp_train <- as.matrix(pp_train)
  for (k in 1:L){
    if (k %in% row.names(pp_train)) {
      pred2[k, a] = pp_train[row.names(pp_train) == k]
    } else {
      pred2[k, a] = NA
    }
  }
}

write.csv(rmse_train, paste0(output_dir, "rmse_train.csv"))
write.csv(r2_train,   paste0(output_dir, "r2_train.csv"))
write.csv(rmse_test,  paste0(output_dir, "rmse_test.csv"))
write.csv(r2_test,    paste0(output_dir, "r2_test.csv"))
write.csv(pred,       paste0(output_dir, "pred_test.csv"))
write.csv(pred2,      paste0(output_dir, "pred_train.csv"))