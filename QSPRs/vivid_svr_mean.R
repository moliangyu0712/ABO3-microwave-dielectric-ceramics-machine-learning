




library(vivid)
library(e1071)
library(ggplot2)
library(colorspace)
library(grid)


windowsFonts(TimesNewRoman = windowsFont("Times New Roman"))

font_family <- "TimesNewRoman"


data <- read.csv(
  "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/targ~ppv+d+pfi+enmP.csv",
  encoding = "UTF-8"
)

L <- dim(data)[2]


predict_svr <- function(model, newdata, ...) {
  predict(model, newdata = newdata)
}


VIVI_SVR <- function(a, train_data, model) {
  set.seed(a)
  
  VIVI_svr <- vivi(
    data = train_data,
    fit = model,
    response = "targ",
    
    reorder = FALSE,
    
    normalized = FALSE,
    importanceType = "agnostic",
    gridSize = 50,
    nmax = 500,
    class = NULL,
    predictFun = predict_svr,
    numPerm = 4
  )
  
  return(VIVI_svr)
}

svr_model <- function(a, traindata, gamma, cost) {
  set.seed(a)
  
  svr <- svm(
    targ ~ .,
    data = traindata,
    gamma = gamma,
    cost = cost,
    kernel = "radial"
  )
  
  return(svr)
}


seed_vec <- 1:1000

VIVIsvr_list <- vector("list", length(seed_vec))

for (i in seq_along(seed_vec)) {
  seed_i <- seed_vec[i]
  
  set.seed(seed_i)
  
  SVR_i <- svr_model(
    a = seed_i,
    traindata = data,
    gamma = 1,
    cost = 16
  )
  
  VIVIsvr_i <- VIVI_SVR(
    a = seed_i,
    train_data = data,
    model = SVR_i
  )
  
  VIVIsvr_i <- as.matrix(VIVIsvr_i)
  
  if (i == 1) {
    feature_order <- colnames(VIVIsvr_i)
    
    if (is.null(feature_order)) {
      stop("VIVIsvr Matrix has no column names, cannot safely align by feature name for averaging.")
    }
  }
  
  VIVIsvr_i <- VIVIsvr_i[feature_order, feature_order]
  
  VIVIsvr_list[[i]] <- VIVIsvr_i
  
  if (i %% 50 == 0) {
    message("Completed  ", i, " / ", length(seed_vec), "  VIVIsvr Computing ")
  }
}

VIVIsvr <- Reduce("+", VIVIsvr_list) / length(VIVIsvr_list)


imp <- diag(VIVIsvr)


ord <- order(imp, decreasing = TRUE)


VIVIsvr_ordered <- VIVIsvr[ord, ord]


importance_result <- data.frame(
  Feature = names(imp)[ord],
  Importance = imp[ord]
)

print(importance_result)


heatmap_guides <- guides(
  fill = guide_colorbar(
    title.position = "top",
    title.hjust = 0.5,
    label.hjust = 0.5,
    barheight = unit(4.5, "cm"),
    barwidth = unit(0.7, "cm"),
    frame.colour = "black",
    ticks.colour = "black",
    title.theme = element_text(
      family = font_family,
      face = "bold",
      margin = margin(b = 20)
    )
  )
)


p_heat <- viviHeatmap(
  mat = VIVIsvr_ordered,
  impPal = rev(colorspace::sequential_hcl(palette = "Red-Blue", n = 100))
) +
  heatmap_guides +
  theme(
    text = element_text(
      family = font_family,
      size = 24
    ),
    
    axis.text.x = element_text(
      size = 24,
      color = "black",
      family = font_family,
      face = "bold",
      angle = 45,
      hjust = 1,
      vjust = 1
    ),
    axis.text.y = element_text(
      size = 24,
      color = "black",
      family = font_family,
      face = "bold"
    ),
    axis.title.x = element_text(
      size = 28,
      family = font_family,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 28,
      family = font_family,
      face = "bold"
    ),
    
    legend.title = element_text(
      size = 24,
      family = font_family,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 20)
    ),
    legend.text = element_text(
      size = 22,
      family = font_family,
      color = "black"
    ),
    
    legend.box = "vertical",
    legend.box.just = "center",
    legend.spacing.y = unit(36, "pt"),
    legend.box.spacing = unit(20, "pt"),
    legend.box.margin = margin(15, 15, 15, 15),
    legend.margin = margin(10, 10, 10, 10),
    
    legend.key.height = unit(1.4, "cm"),
    legend.key.width = unit(0.7, "cm")
  ) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0))

p_heat

ggsave(
  filename = "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/svr_viviNetworkHeatmap_1.png",
  plot = p_heat,
  width = 13,
  height = 11,
  dpi = 300,
  bg = "white"
)


p <- viviNetwork(
  mat = VIVIsvr_ordered
)


for (i in seq_along(p$layers)) {
  if (
    "GeomText" %in% class(p$layers[[i]]$geom) ||
    "GeomLabel" %in% class(p$layers[[i]]$geom)
  ) {
    p$layers[[i]]$aes_params$size <- 15
    p$layers[[i]]$aes_params$family <- font_family
    p$layers[[i]]$aes_params$fontface <- "bold"
  }
}


network_guides <- guides(
  fill = guide_colorbar(
    title.position = "top",
    title.hjust = 0.5,
    label.hjust = 0.5,
    barheight = unit(4.0, "cm"),
    barwidth = unit(0.7, "cm"),
    order = 1,
    frame.colour = "black",
    ticks.colour = "black",
    title.theme = element_text(
      family = font_family,
      face = "bold",
      margin = margin(b = 8)
    )
  ),
  
  colour = guide_colorbar(
    title.position = "top",
    title.hjust = 0.5,
    label.hjust = 0.5,
    barheight = unit(4.0, "cm"),
    barwidth = unit(0.7, "cm"),
    order = 2,
    frame.colour = "black",
    ticks.colour = "black",
    title.theme = element_text(
      family = font_family,
      face = "bold",
      margin = margin(b = 8)
    )
  ),
  
  size = "none",
  linewidth = "none",
  alpha = "none"
)


p <- p +
  network_guides +
  theme(
    text = element_text(
      family = font_family,
      size = 24
    ),
    
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    
    plot.title = element_text(
      size = 30,
      family = font_family,
      face = "bold",
      hjust = 0.5
    ),
    
    legend.title = element_text(
      size = 24,
      family = font_family,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 8)
    ),
    legend.text = element_text(
      size = 22,
      family = font_family,
      color = "black"
    ),
    
    legend.box = "vertical",
    legend.box.just = "center",
    
    legend.spacing.y = unit(8, "pt"),
    legend.box.spacing = unit(8, "pt"),
    legend.box.margin = margin(5, 5, 5, 5),
    legend.margin = margin(3, 3, 3, 3),
    
    legend.key.height = unit(1.0, "cm"),
    legend.key.width = unit(0.7, "cm")
  )

p

ggsave(
  filename = "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/svr_viviNetworkNetwork plot_1.png",
  plot = p,
  width = 13,
  height = 11,
  dpi = 300,
  bg = "white"
)