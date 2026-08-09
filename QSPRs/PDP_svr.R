


library(vivid)
library(e1071)
library(ggplot2)
library(colorspace)
library(grid)
library(scales)
library(zenplots)
library(pdp)

windowsFonts(TimesNewRoman = windowsFont("Times New Roman"))

font_family <- "TimesNewRoman"

data_path <- "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/targ~ppv+d+pfi+enmP.csv"

data <- read.csv(
  data_path,
  encoding = "UTF-8"
)

L <- dim(data)[2]

output_dir <- "D:/MLY/ABO3/Machine learningML/valid_QSPR/Qf_targ~ppv+d+pfi+enmP/PDP_2D"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

predict_svr <- function(model, newdata, ...) {
  predict(model, newdata = newdata)
}

Zen_SVR <- function(a, train_data, model) {
  set.seed(a)
  
  pdp_svr <- pdpZen(
    data = train_data,
    fit = model,
    response = "targ",
    gridSize = 10,
    nmax = 500
  )
  
  return(pdp_svr)
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

seed <- 42

set.seed(seed)

SVR <- svr_model(seed, data, gamma = 1, cost = 16)

Zensvr <- Zen_SVR(seed, data, SVR)

print(Zensvr)


legend_title <- "Predicted targ"

feature_cols <- setdiff(colnames(data), "targ")

feature_cols <- feature_cols[sapply(data[feature_cols], is.numeric)]

feature_pairs <- combn(feature_cols, 2, simplify = FALSE)

pdp_data_list <- list()

for (pair in feature_pairs) {
  
  var_x <- pair[1]
  var_y <- pair[2]
  
  pdp_name <- paste0(var_x, "_", var_y)
  
  message("Computing 2D  PDP Data: ", var_x, " × ", var_y)
  
  pdp_data <- partial(
    object = SVR,
    pred.var = c(var_x, var_y),
    train = data,
    pred.fun = function(object, newdata) {
      predict(object, newdata = newdata)
    },
    grid.resolution = 100
  )
  
  if (!("yhat" %in% colnames(pdp_data))) {
    stop("PDP No  yhat  columns, please check partial()  output structure.")
  }
  
  pdp_data_list[[pdp_name]] <- pdp_data
}

all_yhat <- unlist(
  lapply(pdp_data_list, function(x) x$yhat),
  use.names = FALSE
)

global_lims <- quantile(
  all_yhat,
  probs = c(0.02, 0.98),
  na.rm = TRUE
)


message(
  "Unified color map range: ",
  round(global_lims[1], 4),
  "  to  ",
  round(global_lims[2], 4)
)

plot_2d_pdp <- function(pdp_data,
                        var_x,
                        var_y,
                        fill_limits,
                        legend_title = "Predicted targ",
                        font_family = "TimesNewRoman") {
  
  x_vals <- sort(unique(pdp_data[[var_x]]))
  y_vals <- sort(unique(pdp_data[[var_y]]))
  
  if (length(x_vals) < 2 || length(y_vals) < 2) {
    stop(paste0("Variable combination ", var_x, " × ", var_y, "  has too few unique values, cannot draw 2D  PDP."))
  }
  
  x_step <- min(diff(x_vals))
  y_step <- min(diff(y_vals))
  
  x_tile_range <- range(pdp_data[[var_x]], na.rm = TRUE) + c(-0.5, 0.5) * x_step
  y_tile_range <- range(pdp_data[[var_y]], na.rm = TRUE) + c(-0.5, 0.5) * y_step
  
  border_pad_x <- 0.08 * x_step
  border_pad_y <- 0.08 * y_step
  
  x_border_range <- x_tile_range + c(-border_pad_x, border_pad_x)
  y_border_range <- y_tile_range + c(-border_pad_y, border_pad_y)
  
  axis_pad_x <- 0.08 * x_step
  axis_pad_y <- 0.08 * y_step
  
  p <- ggplot(
    pdp_data,
    aes(
      x = .data[[var_x]],
      y = .data[[var_y]],
      fill = yhat
    )
  ) +
    
    geom_tile(
      width = x_step,
      height = y_step
    ) +
    
    geom_segment(
      aes(
        x = x_border_range[1],
        xend = x_border_range[2],
        y = y_border_range[1],
        yend = y_border_range[1]
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.6,
      lineend = "butt"
    ) +
    
    geom_segment(
      aes(
        x = x_border_range[1],
        xend = x_border_range[2],
        y = y_border_range[2],
        yend = y_border_range[2]
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.6,
      lineend = "butt"
    ) +
    
    geom_segment(
      aes(
        x = x_border_range[1],
        xend = x_border_range[1],
        y = y_border_range[1],
        yend = y_border_range[2]
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.6,
      lineend = "butt"
    ) +
    
    geom_segment(
      aes(
        x = x_border_range[2],
        xend = x_border_range[2],
        y = y_border_range[1],
        yend = y_border_range[2]
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.6,
      lineend = "butt"
    ) +
    
    scale_fill_gradientn(
      name = legend_title,
      colours = c("#313695", "#74ADD1", "#FFFFBF", "#F46D43", "#A50026"),
      limits = fill_limits,
      oob = scales::squish,
      guide = guide_colourbar(
        title.position = "top",
        title.hjust = 0.5,
        barheight = unit(8, "cm"),
        barwidth = unit(0.6, "cm"),
        frame.colour = "black",
        frame.linewidth = 0.4,
        ticks.colour = "black",
        ticks.linewidth = 0.4
      )
    ) +
    
    scale_x_continuous(
      limits = x_border_range + c(-axis_pad_x, axis_pad_x),
      expand = expansion(mult = 0)
    ) +
    
    scale_y_continuous(
      limits = y_border_range + c(-axis_pad_y, axis_pad_y),
      expand = expansion(mult = 0)
    ) +
    
    labs(
      x = var_x,
      y = var_y
    ) +
    
    theme_classic(base_family = font_family) +
    
    theme(
      text = element_text(family = font_family),
      
      axis.title.x = element_text(
        size = 24,
        color = "black",
        margin = margin(t = 10)
      ),
      axis.title.y = element_text(
        size = 24,
        color = "black",
        margin = margin(r = 10)
      ),
      axis.text.x = element_text(
        size = 20,
        color = "black"
      ),
      axis.text.y = element_text(
        size = 20,
        color = "black"
      ),
      
      axis.line = element_blank(),
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.7
      ),
      axis.ticks.length = unit(0.18, "cm"),
      
      legend.title = element_text(
        size = 20,
        color = "black",
        hjust = 0.5
      ),
      legend.text = element_text(
        size = 20,
        color = "black"
      ),
      legend.key.height = unit(8, "cm"),
      legend.key.width = unit(0.6, "cm"),
      
      panel.background = element_blank(),
      panel.border = element_blank(),
      panel.grid = element_blank(),
      
      plot.background = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    )
  
  return(p)
}

pdp_plot_list <- list()

for (pair in feature_pairs) {
  
  var_x <- pair[1]
  var_y <- pair[2]
  
  pdp_name <- paste0(var_x, "_", var_y)
  plot_name <- paste0("PDP_2D_", var_x, "_", var_y)
  file_name <- paste0(plot_name, ".png")
  
  message("Drawing 2D  PDP  plot: ", var_x, " × ", var_y)
  
  p <- plot_2d_pdp(
    pdp_data = pdp_data_list[[pdp_name]],
    var_x = var_x,
    var_y = var_y,
    fill_limits = global_lims,
    legend_title = legend_title,
    font_family = font_family
  )
  
  pdp_plot_list[[plot_name]] <- p
  
  print(p)
  
  ggsave(
    filename = file.path(output_dir, file_name),
    plot = p,
    width = 8,
    height = 7,
    dpi = 300,
    bg = "white"
  )
}

saveRDS(
  pdp_data_list,
  file = file.path(output_dir, "all_2D_PDP_data.rds")
)

saveRDS(
  pdp_plot_list,
  file = file.path(output_dir, "all_2D_PDP_plots.rds")
)

message("All 2D  PDP  plot saved, path: ", output_dir)




  
  
  
