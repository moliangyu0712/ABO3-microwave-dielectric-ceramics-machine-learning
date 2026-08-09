
suppressPackageStartupMessages({
  library(xgboost)
  library(Matrix)
  library(parallel)
  library(doParallel)
  library(foreach)
  library(dplyr)
})

get_env <- function(name, default) {
  value <- Sys.getenv(name)
  if (value == "") {
    return(default)
  }
  return(value)
}

parse_numeric_vector <- function(text) {
  text <- gsub("\\[|\\]|c\\(|\\)|\\s", "", text)
  values <- unlist(strsplit(text, ","))
  values <- values[values != ""]
  as.numeric(values)
}

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("✅ Created directory: ", dir_path, "\n")
  }
}

rmse_fun <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

project_root <- get_env("PROJECT_ROOT", normalizePath(file.path(getwd(), ".."), mustWork = FALSE))
data_dir <- get_env("DATA_DIR", file.path(project_root, "data"))
output_dir <- get_env("OUTPUT_DIR", file.path(project_root, "output", "S2-M1-importance"))

target <- get_env("TARGET", "targ")

train_ratio <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
test_ratio <- 1 - train_ratio

n_resampling <- as.integer(get_env("N_RESAMPLING", "1000"))

xgb_tuning_repeats <- as.integer(get_env("XGB_TUNING_REPEATS", "50"))

max_depth_values <- parse_numeric_vector(get_env("XGB_MAX_DEPTH_VALUES", "2,4,6,8,10"))
learning_rate_values <- parse_numeric_vector(get_env("XGB_LEARNING_RATE_VALUES", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"))

max_depth_values <- as.integer(max_depth_values)

max_depth_L <- length(max_depth_values)
lr_L <- length(learning_rate_values)

xgb_nround_fixed <- as.integer(get_env("XGB_NROUND_FIXED", "25"))

xgb_ncores <- as.integer(get_env("XGB_NTHREAD", "1"))

create_dir_if_not_exists(output_dir)

xgb_result_path <- file.path(output_dir, "hyperparameter_optimization", "xgb", "xgb_result")
create_dir_if_not_exists(xgb_result_path)

cat("========== XGB Parameters ==========\n")
cat("project_root:", project_root, "\n")
cat("data_dir:", data_dir, "\n")
cat("output_dir:", output_dir, "\n")
cat("target:", target, "\n")
cat("train_ratio:", train_ratio, "\n")
cat("test_ratio:", test_ratio, "\n")
cat("n_resampling:", n_resampling, "\n")
cat("xgb_tuning_repeats:", xgb_tuning_repeats, "\n")
cat("max_depth_values:", paste(max_depth_values, collapse = ","), "\n")
cat("learning_rate_values:", paste(learning_rate_values, collapse = ","), "\n")
cat("xgb_nround_fixed:", xgb_nround_fixed, "\n")
cat("xgb_ncores (R parallel):", xgb_ncores, "\n")

data_path_1 <- file.path(data_dir, "data_after-pcc.csv")
data_path_2 <- file.path(data_dir, "data.csv")

if (file.exists(data_path_1)) {
  data_path <- data_path_1
} else if (file.exists(data_path_2)) {
  data_path <- data_path_2
} else {
  stop(paste0("❌ Input data not found: \n", data_path_1, "\n", data_path_2))
}

cat("📂 XGB Reading data: ", data_path, "\n")

data <- read.csv(data_path, encoding = "UTF-8", check.names = TRUE)

if (!(target %in% colnames(data))) {
  stop(paste0("❌ Target variable ", target, "  does not exist!"))
}

feature_names <- setdiff(colnames(data), target)
n_features <- length(feature_names)

cat("✅ Data loaded, samples: ", nrow(data), ", features: ", n_features, "\n")

formula_xgb <- as.formula(paste(target, "~ . - 1"))

if (xgb_ncores > 1) {
  cl <- makeCluster(xgb_ncores)
  registerDoParallel(cl)
  cat("⚙️ Registered parallel cluster cores: ", xgb_ncores, "\n")
} else {
  registerDoSEQ()
}

cat("\n========== Start XGB Hyperparameter tuning ==========\n")

avg_rmse_train <- matrix(0, nrow = max_depth_L, ncol = lr_L)
avg_r2_train <- matrix(0, nrow = max_depth_L, ncol = lr_L)
avg_rmse_test <- matrix(0, nrow = max_depth_L, ncol = lr_L)
avg_r2_test <- matrix(0, nrow = max_depth_L, ncol = lr_L)

start_tuning <- Sys.time()

tuning_results <- foreach(tt = seq_len(xgb_tuning_repeats), .packages = c("xgboost", "Matrix", "dplyr")) %dopar% {
  aver1_tr <- matrix(0, nrow = max_depth_L, ncol = lr_L)
  aver2_tr <- matrix(0, nrow = max_depth_L, ncol = lr_L)
  aver1_te <- matrix(0, nrow = max_depth_L, ncol = lr_L)
  aver2_te <- matrix(0, nrow = max_depth_L, ncol = lr_L)

  nn <- 1
  for (max_depth in max_depth_values) {
    mm <- 1
    for (learning_rate in learning_rate_values) {
      rmse_train <- numeric(10)
      r2_train <- numeric(10)
      rmse_test <- numeric(10)
      r2_test <- numeric(10)

      set.seed(tt)

      par1 <- sample(
        x = 2,
        size = nrow(data),
        replace = TRUE,
        prob = c(train_ratio, test_ratio)
      )

      train1 <- data[par1 == 1, ]

      folds <- cut(seq_len(nrow(train1)), breaks = 10, labels = FALSE)

      for (t in 1:10) {
        train <- train1[folds != t, ]
        test <- train1[folds == t, ]

        if (nrow(test) < 2 || nrow(train) < 2) {
          rmse_train[t] <- NA
          r2_train[t] <- NA
          rmse_test[t] <- NA
          r2_test[t] <- NA
          next
        }

        train_matrix <- sparse.model.matrix(formula_xgb, data = train)
        test_matrix <- sparse.model.matrix(formula_xgb, data = test)

        dtrain <- xgb.DMatrix(train_matrix, label = train[[target]])
        dtest <- xgb.DMatrix(test_matrix, label = test[[target]])

        params <- list(
          max_depth = max_depth,
          eta = learning_rate,
          objective = "reg:squarederror",
          verbosity = 0,
          nthread = 1        )

        set.seed(tt)

        xgb_model <- xgb.train(
          params = params,
          data = dtrain,
          nrounds = xgb_nround_fixed,
          verbose = 0
        )

        ptrain <- predict(xgb_model, dtrain)
        ptest <- predict(xgb_model, dtest)

        rmse_train[t] <- rmse_fun(train[[target]], ptrain)
        r2_train[t] <- suppressWarnings(cor(train[[target]], ptrain)^2)

        rmse_test[t] <- rmse_fun(test[[target]], ptest)
        r2_test[t] <- suppressWarnings(cor(test[[target]], ptest)^2)
      }

      aver1_tr[nn, mm] <- mean(rmse_train, na.rm = TRUE)
      aver2_tr[nn, mm] <- mean(r2_train, na.rm = TRUE)
      aver1_te[nn, mm] <- mean(rmse_test, na.rm = TRUE)
      aver2_te[nn, mm] <- mean(r2_test, na.rm = TRUE)

      mm <- mm + 1
    }
    nn <- nn + 1
  }

  write.csv(aver1_tr, file.path(xgb_result_path, paste0(tt, "_rmse_train.csv")), row.names = FALSE)
  write.csv(aver2_tr, file.path(xgb_result_path, paste0(tt, "_r2_train.csv")), row.names = FALSE)
  write.csv(aver1_te, file.path(xgb_result_path, paste0(tt, "_rmse_test.csv")), row.names = FALSE)
  write.csv(aver2_te, file.path(xgb_result_path, paste0(tt, "_r2_test.csv")), row.names = FALSE)

  list(
    aver1_tr = aver1_tr,
    aver2_tr = aver2_tr,
    aver1_te = aver1_te,
    aver2_te = aver2_te
  )
}

for (res in tuning_results) {
  avg_rmse_train <- avg_rmse_train + res$aver1_tr
  avg_r2_train <- avg_r2_train + res$aver2_tr
  avg_rmse_test <- avg_rmse_test + res$aver1_te
  avg_r2_test <- avg_r2_test + res$aver2_te
}

avg_rmse_train <- avg_rmse_train / xgb_tuning_repeats
avg_r2_train <- avg_r2_train / xgb_tuning_repeats
avg_rmse_test <- avg_rmse_test / xgb_tuning_repeats
avg_r2_test <- avg_r2_test / xgb_tuning_repeats

write.csv(avg_rmse_train, file.path(xgb_result_path, "avg_rmse_train.csv"), row.names = FALSE)
write.csv(avg_r2_train, file.path(xgb_result_path, "avg_r2_train.csv"), row.names = FALSE)
write.csv(avg_rmse_test, file.path(xgb_result_path, "avg_rmse_test.csv"), row.names = FALSE)
write.csv(avg_r2_test, file.path(xgb_result_path, "avg_r2_test.csv"), row.names = FALSE)

best_r2 <- max(avg_r2_test, na.rm = TRUE)
best_idx <- which(avg_r2_test == best_r2, arr.ind = TRUE)[1, ]

best_max_depth <- max_depth_values[best_idx[1]]
best_learning_rate <- learning_rate_values[best_idx[2]]

end_tuning <- Sys.time()

cat("✅ XGB Hyperparameter tuning done, elapsed: ",
    round(difftime(end_tuning, start_tuning, units = "mins"), 2),
    " min\n")

cat("🏆 XGB OptimalParameters: \n")
cat("best_r2:", best_r2, "\n")
cat("best_max_depth:", best_max_depth, "\n")
cat("best_learning_rate:", best_learning_rate, "\n")

cat("\n========== Start XGB Feature importance resampling ==========\n")

im1 <- data.frame(Feature = feature_names)

start_imp <- Sys.time()

imp_results <- foreach(a = seq_len(n_resampling), .packages = c("xgboost", "Matrix", "dplyr")) %dopar% {
  set.seed(a)

  par <- sample(
    x = 2,
    size = nrow(data),
    replace = TRUE,
    prob = c(train_ratio, test_ratio)
  )

  train <- data[par == 1, ]

  train_matrix <- sparse.model.matrix(formula_xgb, data = train)
  dtrain <- xgb.DMatrix(train_matrix, label = train[[target]])

  params <- list(
    max_depth = best_max_depth,
    eta = best_learning_rate,
    objective = "reg:squarederror",
    verbosity = 0,
    nthread = 1  )

  set.seed(a)

  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = xgb_nround_fixed,
    verbose = 0
  )

  imp <- xgb.importance(
    feature_names = colnames(train_matrix),
    model = xgb_model
  )

  iter_imp <- data.frame(
    Feature = feature_names,
    Importance = 0
  )

  if (nrow(imp) > 0) {
    matched_idx <- match(imp$Feature, iter_imp$Feature)
    iter_imp$Importance[matched_idx] <- imp$Gain
  }

  iter_imp$Importance
}

for (a in seq_len(n_resampling)) {
  im1[[paste0("Iter_", a)]] <- imp_results[[a]]
}

end_imp <- Sys.time()

cat("✅ XGB Importance calculation done, elapsed: ",
    round(difftime(end_imp, start_imp, units = "mins"), 2),
    " min\n")

xgb_im1_path <- file.path(output_dir, "xgb_im1.csv")
write.csv(im1, xgb_im1_path, row.names = FALSE)

cat("💾 XGB Output complete: \n")
cat(xgb_im1_path, "\n")

im1_avg <- data.frame(
  Feature = im1$Feature,
  Avg_Importance = rowMeans(im1[, -1], na.rm = TRUE)
)

im1_avg <- im1_avg[order(im1_avg$Avg_Importance, decreasing = TRUE), ]
rownames(im1_avg) <- NULL

cat("\n📈 XGB Mean feature importance: \n")
print(im1_avg)

if (xgb_ncores > 1) {
  stopCluster(cl)
  cat("🔌 XGB Parallel cluster closed\n")
}

cat("🎉 XGB All done!\n")