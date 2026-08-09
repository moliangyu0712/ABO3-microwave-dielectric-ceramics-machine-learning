
suppressPackageStartupMessages({
  library(gbm)
})

get_env <- function(name, default) {
  value <- Sys.getenv(name)
  if (value == "") {
    return(default)
  }
  return(value)
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

gbdt_shrinkage <- as.numeric(get_env("GBDT_SHRINKAGE", "0.01"))
gbdt_n_trees <- as.integer(get_env("GBDT_N_TREES", "3000"))
gbdt_interaction_depth <- as.integer(get_env("GBDT_INTERACTION_DEPTH", "3"))
gbdt_bag_fraction <- as.numeric(get_env("GBDT_BAG_FRACTION", "0.5"))
gbdt_train_fraction <- as.numeric(get_env("GBDT_TRAIN_FRACTION", "0.8"))
gbdt_minobsinnode <- as.integer(get_env("GBDT_MINOBSINNODE", "3"))
gbdt_cv_folds <- as.integer(get_env("GBDT_CV_FOLDS", "10"))

gbdt_n_cores <- as.integer(get_env("GBDT_N_CORES", "1"))

create_dir_if_not_exists(output_dir)

cat("========== GBDT Parameters ==========\n")
cat("project_root:", project_root, "\n")
cat("data_dir:", data_dir, "\n")
cat("output_dir:", output_dir, "\n")
cat("target:", target, "\n")
cat("train_ratio:", train_ratio, "\n")
cat("test_ratio:", test_ratio, "\n")
cat("n_resampling:", n_resampling, "\n")
cat("gbdt_shrinkage:", gbdt_shrinkage, "\n")
cat("gbdt_n_trees:", gbdt_n_trees, "\n")
cat("gbdt_n_cores:", gbdt_n_cores, "\n")

data_path_1 <- file.path(data_dir, "data_after-pcc.csv")
data_path_2 <- file.path(data_dir, "data.csv")

if (file.exists(data_path_1)) {
  data_path <- data_path_1
} else if (file.exists(data_path_2)) {
  data_path <- data_path_2
} else {
  stop(paste0("❌ Input data not found: \n", data_path_1, "\n", data_path_2))
}

cat("📂 GBDT Reading data: ", data_path, "\n")

data <- read.csv(data_path, encoding = "UTF-8", check.names = TRUE)

if (!(target %in% colnames(data))) {
  stop(paste0("❌ Target variable ", target, "  does not exist!"))
}

L <- length(data)
feature_names <- setdiff(colnames(data), target)
n_features <- length(feature_names)

cat("✅ Data loaded, samples: ", nrow(data), ", features: ", n_features, "\n")

im1 <- matrix(0, nrow = n_features, ncol = n_resampling)
im2 <- matrix("", nrow = n_features, ncol = 1)

rownames(im1) <- feature_names
rownames(im2) <- feature_names

formula_gbdt <- as.formula(paste(target, "~ ."))

cat("🚀 Start GBDT Feature importance resampling...\n")
start_time <- Sys.time()

for (a in seq_len(n_resampling)) {
  set.seed(a)

  par <- sample(
    x = 2,
    size = nrow(data),
    replace = TRUE,
    prob = c(train_ratio, test_ratio)
  )

  train <- data[par == 1, ]

  set.seed(a)

  gbm1 <- gbm(
    formula = formula_gbdt,
    data = train,
    var.monotone = rep(0, n_features),
    distribution = "gaussian",
    n.trees = gbdt_n_trees,
    shrinkage = gbdt_shrinkage,
    interaction.depth = gbdt_interaction_depth,
    bag.fraction = gbdt_bag_fraction,
    train.fraction = gbdt_train_fraction,
    n.minobsinnode = gbdt_minobsinnode,
    cv.folds = gbdt_cv_folds,
    keep.data = TRUE,
    verbose = FALSE,
    n.cores = gbdt_n_cores
  )

  best.iter <- gbm.perf(gbm1, method = "cv", plot.it = FALSE)

  gbdt_sum <- summary(
    gbm1,
    n.trees = best.iter,
    plotit = FALSE,
    order = FALSE
  )

  rel_inf_named <- setNames(gbdt_sum$rel.inf, gbdt_sum$var)
  im1[, a] <- rel_inf_named[feature_names]
  im1[is.na(im1[, a]), a] <- 0

  if (a %% 10 == 0 || a == n_resampling) {
    cat("GBDT Completed: ", a, "/", n_resampling, "\n")
  }
}

im2[, 1] <- feature_names

end_time <- Sys.time()
cat("✅ GBDT Importance calculation done, elapsed: ",
    round(difftime(end_time, start_time, units = "mins"), 2),
    " min\n")

gbdt_im1_path <- file.path(output_dir, "gbdt_im1.csv")
gbdt_im2_path <- file.path(output_dir, "gbdt_im2.csv")

write.csv(im1, gbdt_im1_path, row.names = TRUE)
write.csv(im2, gbdt_im2_path, row.names = TRUE)

cat("💾 GBDT Output complete: \n")
cat(gbdt_im1_path, "\n")
cat(gbdt_im2_path, "\n")
cat("🎉 GBDT All done!\n")