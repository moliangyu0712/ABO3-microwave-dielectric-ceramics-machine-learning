library(Metrics)
library(gbm)
library(foreach)
library(doParallel)

get_env <- function(name, default) {
  val <- Sys.getenv(name)
  if (val == "") return(default)
  return(val)
}

project_root       <- get_env("PROJECT_ROOT", "D:/MLY/ABO3/code/machine_learning")
s3_output_dir      <- get_env("S3_OUTPUT_DIR", file.path(project_root, "output/S3-feature_selection"))
gbdt_hp_output_dir <- get_env("GBDT_HP_OUTPUT_DIR", file.path(s3_output_dir, "gbdt/hp"))

target_col         <- get_env("TARGET", "targ")
train_ratio        <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
top_n              <- as.numeric(get_env("TOP_N_FEATURES", "10"))
core               <- as.numeric(get_env("GBDT_N_CORES", "10"))

cv_folds           <- as.numeric(get_env("GBDT_CV_FOLDS", "10"))
cv_repeats         <- as.numeric(get_env("GBDT_CV_REPEATS", "50"))

shrinkage_str      <- get_env("GBDT_SHRINKAGE_VALUES", "0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1")
ntree_str          <- get_env("GBDT_NTREE_VALUES", "100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000")

shrinkage_list     <- as.numeric(strsplit(shrinkage_str, ",")[[1]])
ntree_list         <- as.numeric(strsplit(ntree_str, ",")[[1]])

input_file         <- file.path(s3_output_dir, paste0("selected_top_", top_n, "_features.csv"))
feature_file       <- file.path(s3_output_dir, "feature_combination.csv")

if (!endsWith(gbdt_hp_output_dir, "/")) {
  gbdt_hp_output_dir <- paste0(gbdt_hp_output_dir, "/")
}

cl <- makeCluster(core)
registerDoParallel(cl)

if (!file.exists(input_file)) {
  stop(paste("Input dataset not found: ", input_file))
}
data <- read.csv(input_file, encoding = "UTF-8")

if (!file.exists(feature_file)) {
  stop(paste("Feature combination file not found: ", feature_file))
}
feature <- read.csv(feature_file, encoding = "UTF-8")
names(feature)[1] <- "feature"
L <- dim(feature)[1]

prob_val <- c(train_ratio, 1 - train_ratio)
nrow_val <- length(shrinkage_list)
ncol_val <- length(ntree_list)

foreach(tt = 1:cv_repeats, .packages = c("Metrics", "gbm"), .inorder = FALSE) %dopar% {
  
  for (b in 1:L) {
    name <- feature[b, 1]
    
    file_check1 <- paste(gbdt_hp_output_dir, tt, name, "rmse_train.csv")
    file_check2 <- paste(gbdt_hp_output_dir, tt, name, "r2_train.csv")
    file_check3 <- paste(gbdt_hp_output_dir, tt, name, "rmse_test.csv")
    file_check4 <- paste(gbdt_hp_output_dir, tt, name, "r2_test.csv")
    
    if (file.exists(file_check1) && file.exists(file_check2) && 
        file.exists(file_check3) && file.exists(file_check4)) {
      next
    }
    
    aver1_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver2_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver1_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver2_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    
    formula_str = as.character(feature[b, 1])
    formula <- as.formula(formula_str)
    LL <- length(strsplit(formula_str, "\\+")[[1]])
    
    set.seed(tt)
    par1 <- sample(2, nrow(data), replace = TRUE, prob = prob_val)
    train1 <- data[par1 == 1, ]
    
    folds <- cut(seq_len(nrow(train1)), breaks = cv_folds, labels = FALSE)
    
    mm <- 1    for (shrinkage in shrinkage_list) {
      
      nn <- 1      for (ntree in ntree_list) {
        
        rmse_train <- matrix(0, nrow = 1, ncol = cv_folds)
        r2_train <- matrix(0, nrow = 1, ncol = cv_folds)
        rmse_test <- matrix(0, nrow = 1, ncol = cv_folds)
        r2_test <- matrix(0, nrow = 1, ncol = cv_folds)
        
        for (t in 1:cv_folds) {
          train <- train1[folds != t, ]
          test  <- train1[folds == t, ]
          
          set.seed(tt)
          gbm1 <- gbm(formula, data = train,                                                                  
                      var.monotone = rep(0, LL),
                      distribution = "gaussian",
                      n.trees = ntree,
                      shrinkage = shrinkage,
                      interaction.depth = 3,
                      bag.fraction = 0.5,
                      train.fraction = 1,
                      n.minobsinnode = 3,
                      keep.data = TRUE,
                      verbose = FALSE,
                      n.cores = 1) 
          
          ptrain <- predict(gbm1, train)
          rmse_train[1, t] <- rmse(train[[target_col]], ptrain)
          
          R2a <- data.frame(ptrain = ptrain, targ_val = train[[target_col]])
          la <- lm(targ_val ~ ptrain, data = R2a)
          r2_train[1, t] <- as.numeric(summary(la)$r.squared)
          
          ptest <- predict(gbm1, test)
          rmse_test[1, t] <- rmse(test[[target_col]], ptest)
          
          R2b <- data.frame(ptest = ptest, targ_val = test[[target_col]])
          lb <- lm(targ_val ~ ptest, data = R2b)
          r2_test[1, t] <- as.numeric(summary(lb)$r.squared)
        }
        
        aver1_tr[mm, nn] <- mean(rmse_train)
        aver2_tr[mm, nn] <- mean(r2_train)
        aver1_te[mm, nn] <- mean(rmse_test)
        aver2_te[mm, nn] <- mean(r2_test)
        
        nn <- nn + 1
      }
      mm <- mm + 1
    }
    
    write.csv(aver1_tr, paste(gbdt_hp_output_dir, tt, name, "rmse_train.csv"), row.names = FALSE)
    write.csv(aver2_tr, paste(gbdt_hp_output_dir, tt, name, "r2_train.csv"), row.names = FALSE)
    write.csv(aver1_te, paste(gbdt_hp_output_dir, tt, name, "rmse_test.csv"), row.names = FALSE)
    write.csv(aver2_te, paste(gbdt_hp_output_dir, tt, name, "r2_test.csv"), row.names = FALSE)
  }
}

stopCluster(cl)