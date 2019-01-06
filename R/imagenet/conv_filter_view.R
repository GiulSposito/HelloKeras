library(keras)
library(grid)

# you’ll build a loss function that maximizes the value of a given
# filter in a given convolution layer, and then you’ll use stochastic gradient descent to
# adjust the values of the input image so as to maximize this activation value

# access to keras (mid-level API)
K <- backend()

# create a imagenet network
model <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE
)

# check the model
summary(model)

# select one layer to get the outpu
layer_name <- "block3_conv1"
filter_index <- 1

# get access to layer
model_layer <- get_layer(model, layer_name)
layer_output <- model_layer$output #output tensor

# build a 'loss function'
loss <- K$mean(layer_output[,,,filter_index]) # tensor operation

# To implement gradient descent, you’ll need the gradient of this loss with respect to
# the model’s input. To do this, you’ll use the function packaged with the gradients
# backend module of Keras.

grads <- K$gradients(loss, model$input)[[1]]

# normalization of gradients
grads <- grads / (K$sqrt(K$mean(K$square(grads))) + 1e-5)

# Now you need a way to compute the value of the loss tensor and the gradient tensor,
# given an input image. You can define a Keras backend function to do this: is a iterate
# function that takes a tensor (as a list of tensors of size 1) and returns a list of two tensors:
# the loss value and the gradient value.

iterate <- K$`function`(list(model$input), list(loss, grads))

c(loss_value, grads_value) %<-% iterate(list(array(0,dim=c(1,150,150,3))))

# At this point, you can define an R loop to do stochastic gradient ascent.

input_img_data <-   array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128

step <- 1
for (i in 1:40) {
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  input_img_data <- input_img_data + (grads_value * step)
}

deprocess_image <- function(x) {
  dms <- dim(x)
  x <- x - mean(x)
  x <- x / (sd(x)+1e5)
  x <- x * 0.1
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  array(x, dim=dms)
}

input_img_data %>%
  .[1,,,] %>% 
  deprocess_image() %>%
  grid.raster()

generate_pattern <- function(layer_name, filter_index, size = 150) {
  layer_output <- model$get_layer(layer_name)$output
  loss <- K$mean(layer_output[,,,filter_index])
  grads <- K$gradients(loss, model$input)[[1]]
  grads <- grads / (K$sqrt(K$mean(K$square(grads))) + 1e-5)
  iterate <- K$`function`(list(model$input), list(loss, grads))
  input_img_data <-
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step)
  }
  img <- input_img_data[1,,,]
  deprocess_image(img)
}


grid.raster(generate_pattern("block3_conv1",1))


library(grid)
library(gridExtra)
dir.create("./data/vgg_filters")

for (layer_name in c("block1_conv1", "block2_conv1","block3_conv1", "block4_conv1")) {
  
  size <- 140
  
  png(paste0("./data/vgg_filters", layer_name, ".png"), width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for (i in 0:7) {
    for (j in 0:7) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern,
                         width = unit(0.9, "npc"),
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }
