





library(vivid)
library(xgboost)
library(Matrix)
library(ggplot2)
library(colorspace)
library(grid)


windowsFonts(TimesNewRoman = windowsFont("Times New Roman"))

font_family <- "TimesNewRoman"



data <- read.csv(
  "D:/MLY/ABO3/Machine learningML/validation/TCF_targ~7/selected_top_7_features.csv",
  encoding = "UTF-8"
)

L <- dim(data)[2]


predict_xgb <- function(model, newdata, ...) {
  sparse_mat <- sparse.model.matrix(targ ~ . -1, data = newdata)
  predict(model, newdata = sparse_mat)
}


VIVI_XGB <- function(a, train_data, model) {
  set.seed(a)
  
  VIVI_xgb <- vivi(
    data = train_data,
    fit = model,
    response = "targ",
    
    reorder = FALSE,
    
    normalized = FALSE,
    importanceType = "agnostic",
    gridSize = 50,
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
  
  xgb <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 25
  )
  
  return(xgb)
}


seed_vec <- 1:1000

VIVIxgb_list <- vector("list", length(seed_vec))

for (i in seq_along(seed_vec)) {
  seed_i <- seed_vec[i]
  
  set.seed(seed_i)
  
  xgb_i <- xgb_model(
    a = seed_i,
    traindata = data,
    max_depth = 6,
    learning_rate = 0.2
  )
  
  VIVIxgb_i <- VIVI_XGB(
    a = seed_i,
    train_data = data,
    model = xgb_i
  )
  
  VIVIxgb_i <- as.matrix(VIVIxgb_i)
  
  if (i == 1) {
    feature_order <- colnames(VIVIxgb_i)
    
    if (is.null(feature_order)) {
      stop("VIVIxgb Matrix has no column names, cannot safely align by feature name for averaging.")
    }
  }
  
  VIVIxgb_i <- VIVIxgb_i[feature_order, feature_order]
  
  VIVIxgb_list[[i]] <- VIVIxgb_i
  
  if (i %% 50 == 0) {
    message("Completed  ", i, " / ", length(seed_vec), "  VIVIxgb Computing ")
  }
}

VIVIxgb <- Reduce("+", VIVIxgb_list) / length(VIVIxgb_list)


imp <- diag(VIVIxgb)


ord <- order(imp, decreasing = TRUE)


VIVIxgb_ordered <- VIVIxgb[ord, ord]


importance_result <- data.frame(
  Feature = names(imp)[ord],
  Importance = imp[ord]
)

print(importance_result)


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


p <- viviNetwork(
  mat = VIVIxgb_ordered
)


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