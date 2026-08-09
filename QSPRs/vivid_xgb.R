







library(vivid)


windowsFonts(TimesNewRoman = windowsFont("Times New Roman"))

font_family <- "TimesNewRoman"



data <- read.csv(
  "D:/MLY/ABO3/Machine learningML/validation/TCF_targ~7/selected_top_7_features.csv",
  encoding = "UTF-8")
  
  
  
  
  
  
  
  
  library(vivid)
  
  
  windowsFonts(TimesNewRoman = windowsFont("Times New Roman"))
  
  font_family <- "TimesNewRoman"
  
  
  
  data <- read.csv(
    "D:/MLY/ABO3/Machine learningML/validation/TCF_targ~7/selected_top_7_features.csv",
    encoding = "UTF-8"
  )
  
  L <- dim(data)[2]
  
  
  library(xgboost)
  library(Matrix)
  
  predict_xgb <- function(model, newdata, ...) {
    sparse_mat <- sparse.model.matrix(targ ~ . -1, data = newdata)
    predict(model, newdata = sparse_mat)
  }
  
  
  VIVI_XGB <- function(a, train_data, model) {
    set.seed(a)
    
    VIVI_xgb <- vivi(
      data = train_data,
      fit = model,      response = "targ",
      reorder = TRUE,
      normalized = FALSE,
      importanceType = "agnostic",      gridSize = 50,
      nmax = 500,
      class = NULL,
      predictFun = predict_xgb,
      numPerm = 4
    )
    
    return(VIVI_xgb)
  }
  
  xgb_model <- function(a, traindata, max_depth, learning_rate) {
    set.seed(a)
    
    params <- list(
      max_depth = max_depth,
      learning_rate = learning_rate,
      objective = "reg:squarederror",
      verbosity = 0
    )
    
    train_matrix <- sparse.model.matrix(targ ~ . -1, data = traindata)
    dtrain <- xgb.DMatrix(train_matrix, label = traindata$targ)
    
    xgb <- xgb.train(params, dtrain, nrounds = 25)
    
    return(xgb)
  }
  
  set.seed(42)
  
  xgb <- xgb_model(42, data, 6, 0.2)
  
  VIVIxgb <- VIVI_XGB(42, data, xgb)
  
  library(ggplot2)
  
  
  imp <- diag(VIVIxgb)
  
  
  ord <- order(imp, decreasing = TRUE)
  
  
  VIVIxgb_ordered <- VIVIxgb[ord, ord]
  
  
  data.frame(
    Feature = names(imp)[ord],
    Importance = imp[ord]
  )
  
  
  p_heat <- viviHeatmap(
    mat = VIVIxgb_ordered,
    impPal = rev(colorspace::sequential_hcl(palette = "Red-Blue", n = 100))
  ) +
    theme(
      text = element_text(family = font_family),
      axis.text.x = element_text(
        size = 16,
        color = "black",
        family = font_family,
        face = "bold"
      ),
      axis.text.y = element_text(
        size = 16,
        color = "black",
        family = font_family,
        face = "bold"
      ),
      axis.title.x = element_text(
        size = 18,
        family = font_family,
        face = "bold"
      ),
      axis.title.y = element_text(
        size = 18,
        family = font_family,
        face = "bold"
      ),
      legend.title = element_text(
        size = 16,
        family = font_family,
        face = "bold"
      ),
      legend.text = element_text(
        size = 14,
        family = font_family
      )
    ) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0))
  
  p_heat
  
  ggsave(
    filename = "D:/MLY/ABO3/Machine learningML/validation/TCF_targ~7/xgb_viviNetworkHeatmap.png",
    plot = p_heat,
    width = 10,
    height = 8,
    dpi = 300,
    bg = "white"
  )
  
  
  p <- viviNetwork(mat = VIVIxgb_ordered)
  
  
  for (i in seq_along(p$layers)) {
    if (
      "GeomText" %in% class(p$layers[[i]]$geom) ||
      "GeomLabel" %in% class(p$layers[[i]]$geom)
    ) {
      p$layers[[i]]$aes_params$size <- 7
      p$layers[[i]]$aes_params$family <- font_family
      p$layers[[i]]$aes_params$fontface <- "bold"
    }
  }
  
  
  p <- p +
    theme(
      text = element_text(family = font_family),
      plot.title = element_text(
        size = 22,
        family = font_family,
        face = "bold"
      ),
      axis.text.x = element_text(
        size = 16,
        family = font_family,
        color = "black",
        face = "bold"
      ),
      axis.text.y = element_text(
        size = 16,
        family = font_family,
        color = "black",
        face = "bold"
      ),
      axis.title.x = element_text(
        size = 18,
        family = font_family,
        face = "bold"
      ),
      axis.title.y = element_text(
        size = 18,
        family = font_family,
        face = "bold"
      ),
      legend.title = element_text(
        size = 16,
        family = font_family,
        face = "bold"
      ),
      legend.text = element_text(
        size = 14,
        family = font_family
      )
    )
  
  p
  
  ggsave(
    filename = "D:/MLY/ABO3/Machine learningML/validation/TCF_targ~7/xgb_viviNetworkNetwork plot.png",
    plot = p,
    width = 10,
    height = 8,
    dpi = 300,
    bg = "white"
  )