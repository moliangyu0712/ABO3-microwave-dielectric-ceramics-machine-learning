library(e1071)
library(kernelshap)
library(shapviz)
library(ggplot2)
library(Matrix)
library(grid)


font_family <- "Times New Roman"

bee_base_size <- 20
bee_axis_title_size <- 22
bee_axis_text_size <- 20
bee_legend_title_size <- 20
bee_legend_text_size <- 20

dep_base_size <- 16
dep_axis_title_size <- 16
dep_axis_text_size <- 16
dep_legend_title_size <- 16
dep_legend_text_size <- 16


data <- read.csv(
  "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/targ~ppv+d+pfi+enmP.csv",
  fileEncoding = "UTF-8"
)

data$targ <- as.numeric(data$targ)

model_svm <- svm(
  targ ~ .,
  data = data,
  gamma = 1,
  cost = 16,
  kernel = "radial"
)

X <- data[, setdiff(names(data), "targ")]
X_explain <- X

set.seed(42)
n_bg <- nrow(X)
idx_bg <- sample(seq_len(nrow(X)), n_bg)
bg_X <- X[idx_bg, ]

pred_fun <- function(object, newdata) {
  as.numeric(predict(object, newdata = newdata))
}

set.seed(42)
ks <- kernelshap(
  object = model_svm,
  X = X_explain,
  bg_X = bg_X,
  pred_fun = pred_fun
)

sv <- shapviz(ks, X = X_explain)


S <- as.matrix(ks$S)

shap_importance <- data.frame(
  feature = colnames(S),
  mean_abs_shap = colMeans(abs(S))
)

shap_importance <- shap_importance[order(-shap_importance$mean_abs_shap), ]

print(shap_importance)


shap_importance_bar <- shap_importance[order(shap_importance$mean_abs_shap), ]

x_max <- max(shap_importance_bar$mean_abs_shap) * 1.15

p_bar <- ggplot(
  shap_importance_bar,
  aes(
    x = mean_abs_shap,
    y = reorder(feature, mean_abs_shap)
  )
) +
  geom_col(
    width = 0.4,
    fill = "#2E86AB"
  ) +
  scale_x_continuous(
    limits = c(0, x_max),
    expand = c(0, 0)
  ) +
  theme_bw() +
  theme(
    axis.title.x = element_text(
      size = 16,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 16,
      face = "bold"
    ),
    axis.text.x = element_text(
      size = 14,
      color = "black"
    ),
    axis.text.y = element_text(
      size = 14,
      color = "black"
    ),
    plot.title = element_text(
      size = 18,
      face = "bold",
      hjust = 0.5
    ),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "Mean |SHAP value|",
    y = "Feature",
    title = "SHAP Feature Importance"
  )

p_bar

ggsave(
  filename = "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/SHAP_importance_kernelshap.png",
  plot = p_bar,
  width = 5,
  height = 4,
  dpi = 300,
  type = "cairo"
)


p_bee <- sv_importance(
  sv,
  kind = "beeswarm"
)

point_layer <- 2

p_bee$layers[[point_layer]]$aes_params$shape <- 21
p_bee$layers[[point_layer]]$aes_params$size <- 2.5
p_bee$layers[[point_layer]]$aes_params$stroke <- 0.35

p_bee$layers[[point_layer]]$mapping$fill <- p_bee$layers[[point_layer]]$mapping$colour

p_bee_final <- p_bee +
  scale_color_gradient(
    low = "#FFCC33",
    high = "#6600CC",
    name = "Feature value"
  ) +
  scale_fill_gradient(
    low = scales::alpha("#FFCC33", 0.75),
    high = scales::alpha("#6600CC", 0.75),
    name = "Feature value"
  ) +
  labs(
    x = "SHAP Value",
    y = "Features"
  ) +
  theme_bw(
    base_family = font_family,
    base_size = bee_base_size
  ) +
  theme(
    text = element_text(
      family = font_family
    ),
    panel.grid = element_blank(),
    axis.title.x = element_text(
      size = bee_axis_title_size,
      face = "bold",
      family = font_family
    ),
    axis.title.y = element_text(
      size = bee_axis_title_size,
      face = "bold",
      family = font_family
    ),
    axis.text.x = element_text(
      size = bee_axis_text_size,
      color = "black",
      family = font_family
    ),
    axis.text.y = element_text(
      size = bee_axis_text_size,
      color = "black",
      family = font_family
    ),
    legend.position = "right",
    legend.title = element_text(
      size = bee_legend_title_size,
      face = "bold",
      family = font_family
    ),
    legend.text = element_text(
      size = bee_legend_text_size,
      color = "black",
      family = font_family
    ),
    plot.title = element_text(
      size = bee_axis_title_size,
      face = "bold",
      hjust = 0.5,
      family = font_family
    ),
    strip.text = element_text(
      size = bee_axis_text_size,
      face = "bold",
      family = font_family
    )
  ) +
  guides(
    color = "none",
    fill = guide_colorbar(
      title = "Feature value",
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(0.5, "cm"),
      barheight = unit(11, "cm"),
      ticks = TRUE,
      ticks.colour = "black",
      label.position = "right"
    )
  )

p_bee_final

ggsave(
  filename = "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/SHAP_summary_kernelshap.png",
  plot = p_bee_final,
  width = 8,
  height = 6,
  dpi = 300,
  type = "cairo"
)


top_feature <- shap_importance$feature[1]

print(top_feature)

p_dep <- sv_dependence(
  sv,
  v = top_feature,
  color_var = top_feature
)

p_dep$layers[[1]]$aes_params$size <- 2

p_dep_final <- p_dep +
  geom_smooth(
    method = "loess",
    se = FALSE,
    color = "red",
    linewidth = 1.2,
    linetype = "dashed"
  ) +
  scale_color_gradient(
    low = "#FFCC33",
    high = "#6600CC"
  ) +
  labs(
    x = paste0(top_feature, " (Feature value)"),
    y = paste0("SHAP value for ", top_feature),
    color = top_feature
  ) +
  theme_bw(
    base_family = font_family,
    base_size = dep_base_size
  ) +
  theme(
    text = element_text(
      family = font_family
    ),
    panel.grid.major = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    panel.grid.minor = element_line(
      color = "grey95",
      linewidth = 0.2
    ),
    axis.title.x = element_text(
      size = dep_axis_title_size,
      face = "bold",
      family = font_family
    ),
    axis.title.y = element_text(
      size = dep_axis_title_size,
      face = "bold",
      family = font_family
    ),
    axis.text.x = element_text(
      size = dep_axis_text_size,
      color = "black",
      family = font_family
    ),
    axis.text.y = element_text(
      size = dep_axis_text_size,
      color = "black",
      family = font_family
    ),
    legend.position = "right",
    legend.title = element_text(
      size = dep_legend_title_size,
      face = "bold",
      family = font_family
    ),
    legend.text = element_text(
      size = dep_legend_text_size,
      color = "black",
      family = font_family
    ),
    plot.title = element_text(
      size = 18,
      face = "bold",
      hjust = 0.5,
      family = font_family
    ),
    strip.text = element_text(
      size = dep_axis_text_size,
      face = "bold",
      family = font_family
    )
  ) +
  guides(
    color = guide_colorbar(
      title = top_feature,
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(0.5, "cm"),
      barheight = unit(10, "cm"),
      ticks = TRUE,
      ticks.colour = "black",
      label.position = "right"
    )
  )

p_dep_final

ggsave(
  filename = "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/d_SHAP_kernelshap.png",
  plot = p_dep_final,
  width = 8,
  height = 6,
  dpi = 300,
  type = "cairo"
)