library(keras)
library(grid)
library(gridExtra)
library(glue)


deprocess_image <- function(x) {
  
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}


generate_pattern <- function(layer_name, filter_index, size = 150, steps=40) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # This function returns the loss and grads given the input picture
  iterate <- k_function(list(model$input), list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- the.image
    #array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:steps) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img) 
}

model <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE
)

summary(model)

# access to keras (mid-level API)
K <- backend()

for (layer_name in c("block1_conv1", "block2_conv1", 
                     "block3_conv1", "block4_conv1")) {
  size <- 140
  
  print(glue("layer: {layer_name}"))
  
  png(paste0("./images/vgg_filters/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for (i in 0:2) {
    for (j in 0:2) {
      print(glue("filter i:{i} j:{j}"))
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern, 
                         width = unit(0.9, "npc"), 
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }  
  }
  
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}

size <- 150
(array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128) %>%
  deprocess_image() %>% 
  .[1,,,] %>% 
  grid.raster()


generate_pattern("block4_conv1",3*8,140, 10) %>% 
  grid.raster()

the.image <- image_load("./images/jushober.jpg", target_size = c(150,150)) %>% 
  image_to_array() %>% 
  array(dim=c(1,150,150,3))

the.image %>% 
  deprocess_image() %>% 
  .[1,,,] %>% 
  grid.raster()
