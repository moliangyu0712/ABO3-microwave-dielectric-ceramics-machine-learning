
suppressPackageStartupMessages({
  library(Metrics)
  library(randomForest)
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

project_root <- get_env("PROJECT_ROOT", normalizePath(file.path(getwd(), ".."), mustWork = FALSE))
data_dir <- get_env("DATA_DIR", file.path(project_root, "data"))
output_dir <- get_env("OUTPUT_DIR", file.path(project_root, "output", "S2-M1-importance"))

target <- get_env("TARGET", "targ")

train_ratio <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
test_ratio <- 1 - train_ratio

n_resampling <- as.integer(get_env("N_RESAMPLING", "1000"))

rf_tuning_repeats <- as.integer(get_env("RF_TUNING_REPEATS", "50"))

maxnodes_values <- parse_numeric_vector(get_env("RF_MAXNODES_VALUES", "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60"))
ntree_values <- parse_numeric_vector(get_env("RF_NTREE_VALUES", "100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000"))

maxnodes_values <- as.integer(maxnodes_values)
ntree_values <- as.integer(ntree_values)

maxnodes_L <- length(maxnodes_values)
ntree_L <- length(ntree_values)

rf_n_cores <- as.integer(get_env("RF_N_CORES", "1"))

create_dir_if_not_exists(output_dir)

rf_result_path <- file.path(output_dir, "hyperparameter_optimization", "rf", "rf_result")
create_dir_if_not_exists(rf_result_path)

cat("========== RF Parameters ==========\n")
cat("project_root:", project_root, "\n")
cat("data_dir:", data_dir, "\n")
cat("output_dir:", output_dir, "\n")
cat("target:", target, "\n")
cat("train_ratio:", train_ratio, "\n")
cat("test_ratio:", test_ratio, "\n")
cat("n_resampling:", n_resampling, "\n")
cat("rf_tuning_repeats:", rf_tuning_repeats, "\n")
cat("maxnodes_values:", paste(maxnodes_values, collapse = ","), "\n")
cat("ntree_values:", paste(ntree_values, collapse = ","), "\n")
cat("rf_n_cores:", rf_n_cores, "\n")

data_path_1 <- file.path(data_dir, "data_after-pcc.csv")
data_path_2 <- file.path(data_dir, "data.csv")

if (file.exists(data_path_1)) {
  data_path <- data_path_1
} else if (file.exists(data_path_2)) {
  data_path <- data_path_2
} else {
  stop(paste0("❌ Input data not found: \n", data_path_1, "\n", data_path_2))
}

cat("📂 RF Reading data: ", data_path, "\n")

data <- read.csv(data_path, encoding = "UTF-8", check.names = TRUE)

if (!(target %in% colnames(data))) {
  stop(paste0("❌ Target variable ", target, "  does not exist!"))
}

feature_names <- setdiff(colnames(data), target)
n_features <- length(feature_names)

cat("✅ Data loaded, samples: ", nrow(data), ", features: ", n_features, "\n")

formula_rf <- as.formula(paste(target, "~ ."))

if (rf_n_cores > 1) {
  cl <- makeCluster(rf_n_cores)
  registerDoParallel(cl)
  cat("⚙️ Registered parallel cluster cores: ", rf_n_cores, "\n")
} else {
  registerDoSEQ()
}

cat("\n========== Start RF Hyperparameter tuning ==========\n")

avg_rmse_train <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
avg_r2_train <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
avg_rmse_test <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
avg_r2_test <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)

start_tuning <- Sys.time()

tuning_results <- foreach(tt = seq_len(rf_tuning_repeats), .packages = c("randomForest", "Metrics", "dplyr")) %dopar% {
  aver1_tr <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
  aver2_tr <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
  aver1_te <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)
  aver2_te <- matrix(0, nrow = maxnodes_L, ncol = ntree_L)

  mm <- 1
  for (maxnodes in maxnodes_values) {
    nn <- 1
    for (ntree in ntree_values) {
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

        set.seed(tt)

        rf <- randomForest(
          formula = formula_rf,
          data = train,
          ntree = ntree,
          maxnodes = maxnodes,
          importance = FALSE
        )

        ptrain <- predict(rf, train)
        ptest <- predict(rf, test)

        rmse_train[t] <- rmse(train[[target]], ptrain)
        r2_train[t] <- suppressWarnings(cor(train[[target]], ptrain)^2)

        rmse_test[t] <- rmse(test[[target]], ptest)
        r2_test[t] <- suppressWarnings(cor(test[[target]], ptest)^2)
      }

      aver1_tr[mm, nn] <- mean(rmse_train, na.rm = TRUE)
      aver2_tr[mm, nn] <- mean(r2_train, na.rm = TRUE)
      aver1_te[mm, nn] <- mean(rmse_test, na.rm = TRUE)
      aver2_te[mm, nn] <- mean(r2_test, na.rm = TRUE)

      nn <- nn + 1
    }
    mm <- mm + 1
  }
  
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

avg_rmse_train <- avg_rmse_train / rf_tuning_repeats
avg_r2_train <- avg_r2_train / rf_tuning_repeats
avg_rmse_test <- avg_rmse_test / rf_tuning_repeats
avg_r2_test <- avg_r2_test / rf_tuning_repeats

write.csv(avg_rmse_train, file.path(rf_result_path, "rmse_train.csv"), row.names = FALSE)
write.csv(avg_r2_train, file.path(rf_result_path, "r2_train.csv"), row.names = FALSE)
write.csv(avg_rmse_test, file.path(rf_result_path, "rmse_test.csv"), row.names = FALSE)
write.csv(avg_r2_test, file.path(rf_result_path, "r2_test.csv"), row.names = FALSE)

best_r2 <- max(avg_r2_test, na.rm = TRUE)
best_idx <- which(avg_r2_test == best_r2, arr.ind = TRUE)[1, ]

best_maxnodes <- maxnodes_values[best_idx[1]]
best_ntree <- ntree_values[best_idx[2]]

end_tuning <- Sys.time()

cat("✅ RF Hyperparameter tuning done, elapsed: ",
    round(difftime(end_tuning, start_tuning, units = "mins"), 2),
    " min\n")

cat("🏆 RF OptimalParameters: \n")
cat("best_r2:", best_r2, "\n")
cat("best_maxnodes:", best_maxnodes, "\n")
cat("best_ntree:", best_ntree, "\n")

cat("\n========== Start RF Feature importance resampling ==========\n")

im1 <- matrix(0, nrow = n_features, ncol = n_resampling)
im2 <- matrix(0, nrow = n_features, ncol = n_resampling)

rownames(im1) <- feature_names
rownames(im2) <- feature_names

start_imp <- Sys.time()

imp_results <- foreach(b = seq_len(n_resampling), .packages = c("randomForest", "dplyr")) %dopar% {
  set.seed(b)

  par <- sample(
    x = 2,
    size = nrow(data),
    replace = TRUE,
    prob = c(train_ratio, test_ratio)
  )

  train <- data[par == 1, ]

  set.seed(b)

  rf <- randomForest(
    formula = formula_rf,
    data = train,
    ntree = best_ntree,
    maxnodes = best_maxnodes,
    importance = TRUE,
    proximity = FALSE
  )

  imp <- rf$importance

  list(
    im1 = imp[feature_names, 1],
    im2 = imp[feature_names, 2]
  )
}

for (b in seq_len(n_resampling)) {
  im1[, b] <- imp_results[[b]]$im1
  im2[, b] <- imp_results[[b]]$im2
}

end_imp <- Sys.time()

cat("✅ RF Importance calculation done, elapsed: ",
    round(difftime(end_imp, start_imp, units = "mins"), 2),
    " min\n")

rf_im1_path <- file.path(output_dir, "rf_im1.csv")
rf_im2_path <- file.path(output_dir, "rf_im2.csv")

write.csv(im1, rf_im1_path, row.names = TRUE)
write.csv(im2, rf_im2_path, row.names = TRUE)

cat("💾 RF Output complete: \n")
cat(rf_im1_path, "\n")
cat(rf_im2_path, "\n")

cat("\n========== RF Final model validation ==========\n")

set.seed(123)
par <- sample(
  x = 2,
  size = nrow(data),
  replace = TRUE,
  prob = c(train_ratio, test_ratio)
)

train <- data[par == 1, ]
test <- data[par == 2, ]

final_rf <- randomForest(
  formula_rf,
  data = train,
  ntree = best_ntree,
  maxnodes = best_maxnodes,
  importance = TRUE
)

p1 <- predict(final_rf, train)
p2 <- predict(final_rf, test)

cat("Training set R2 =", round(cor(p1, train[[target]])^2, 4), "\n")
cat("Test set R2 =", round(cor(p2, test[[target]])^2, 4), "\n")

if (rf_n_cores > 1) {
  stopCluster(cl)
  cat("🔌 RF Parallel cluster closed\n")
}

cat("🎉 RF All done!\n")