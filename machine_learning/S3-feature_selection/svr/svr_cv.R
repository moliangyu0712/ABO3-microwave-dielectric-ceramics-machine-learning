library(Metrics)
library(e1071)
library(foreach)
library(doParallel)

get_env <- function(name, default) {
  val <- Sys.getenv(name)
  if (val == "") return(default)
  return(val)
}

project_root       <- get_env("PROJECT_ROOT", "D:/MLY/ABO3/code/machine_learning")
s3_output_dir      <- get_env("S3_OUTPUT_DIR", file.path(project_root, "output/S3-feature_selection"))
svr_hp_output_dir  <- get_env("SVR_HP_OUTPUT_DIR", file.path(s3_output_dir, "svr/hp"))

target_col         <- get_env("TARGET", "targ")
train_ratio        <- as.numeric(get_env("TRAIN_RATIO", "0.7"))
top_n              <- as.numeric(get_env("TOP_N_FEATURES", "10"))
core               <- as.numeric(get_env("SVR_N_CORES", "10"))

cv_folds           <- as.numeric(get_env("SVR_CV_FOLDS", "10"))
cv_repeats         <- as.numeric(get_env("SVR_CV_REPEATS", "50"))

gamma_str          <- get_env("SVR_GAMMA_VALUES", "0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256")
cost_str           <- get_env("SVR_COST_VALUES", "0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256")

gamma_list         <- as.numeric(strsplit(gamma_str, ",")[[1]])
cost_list          <- as.numeric(strsplit(cost_str, ",")[[1]])

input_file         <- file.path(s3_output_dir, paste0("selected_top_", top_n, "_features.csv"))
feature_file       <- file.path(s3_output_dir, "feature_combination.csv")

if (!endsWith(svr_hp_output_dir, "/")) {
  svr_hp_output_dir <- paste0(svr_hp_output_dir, "/")
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
nrow_val <- length(gamma_list)
ncol_val <- length(cost_list)

foreach(tt = 1:cv_repeats, .packages = c("Metrics", "e1071"), .inorder = FALSE) %dopar% {
  
  for (b in c(1:L)){
    name <- feature[b, 1]
    
    file_check1 <- paste(svr_hp_output_dir, tt, name, "rmse_train.csv")
    file_check2 <- paste(svr_hp_output_dir, tt, name, "r2_train.csv")
    file_check3 <- paste(svr_hp_output_dir, tt, name, "rmse_test.csv")
    file_check4 <- paste(svr_hp_output_dir, tt, name, "r2_test.csv")
    
    if (file.exists(file_check1) && file.exists(file_check2) && 
        file.exists(file_check3) && file.exists(file_check4)) {
      next
    }
    
    aver1_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)    aver2_tr <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver1_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    aver2_te <- matrix(0, nrow = nrow_val, ncol = ncol_val)
    
    formula_str <- as.character(feature[b, 1])
    formula_obj <- as.formula(formula_str)
    
    set.seed(tt)
    par1 <- sample(2, nrow(data), replace = TRUE, prob = prob_val)
    train1 <- data[par1 == 1, ]
    
    folds <- cut(seq_len(nrow(train1)), breaks = cv_folds, labels = FALSE)
    
    nn <- 1
    for (gamma in gamma_list){
      mm <- 1
      for (cost in cost_list){
        rmse_train <- matrix(0, nrow = 1, ncol = cv_folds)
        r2_train <- matrix(0, nrow = 1, ncol = cv_folds)
        rmse_test <- matrix(0, nrow = 1, ncol = cv_folds)
        r2_test <- matrix(0, nrow = 1, ncol = cv_folds)
        
        for (t in c(1:cv_folds)){
          train <- train1[folds != t, ]
          test  <- train1[folds == t, ]
          
          set.seed(tt)
          svr <- svm(formula_obj, train, gamma = gamma, cost = cost, kernel = "radial")
          
          ptrain <- predict(svr, train)
          rmse_train[1, t] <- rmse(train[[target_col]], ptrain)
          
          R2a <- data.frame(ptrain = ptrain, targ_val = train[[target_col]])
          la <- lm(targ_val ~ ptrain, data = R2a)
          r2_train[1, t] <- summary(la)$r.squared
          
          ptest <- predict(svr, test)
          rmse_test[1, t] <- rmse(test[[target_col]], ptest)
          
          R2b <- data.frame(ptest = ptest, targ_val = test[[target_col]])
          lb <- lm(targ_val ~ ptest, data = R2b)
          r2_test[1, t] <- summary(lb)$r.squared
        }
        
        aver1_tr[nn, mm] <- mean(rmse_train[1, ])
        aver2_tr[nn, mm] <- mean(r2_train[1, ])
        aver1_te[nn, mm] <- mean(rmse_test[1, ])
        aver2_te[nn, mm] <- mean(r2_test[1, ])
        
        mm <- mm + 1
      }
      nn <- nn + 1
    }
    
    write.csv(aver1_tr, paste(svr_hp_output_dir, tt, name, "rmse_train.csv"))
    write.csv(aver2_tr, paste(svr_hp_output_dir, tt, name, "r2_train.csv"))
    write.csv(aver1_te, paste(svr_hp_output_dir, tt, name, "rmse_test.csv"))
    write.csv(aver2_te, paste(svr_hp_output_dir, tt, name, "r2_test.csv"))
  }
}

stopCluster(cl)