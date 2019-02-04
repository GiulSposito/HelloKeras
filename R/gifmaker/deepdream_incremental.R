library(keras)
library(glue)

# disable training operations
k_set_learning_phase(0)

# import model 
model <- application_inception_v3(
  weights = "imagenet",
  include_top = F #without classification layers
)


# Named mapping layer names to a coefficient
# quantifying how much the layer's activation
# will contribute to the loss we will seek to maximize.
# Note that these are layer names as they appear
# in the built-in InceptionV3 application.
# You can list all layer names using `summary(model)`.
layer_contributions <- list(
  mixed2 = 0.2,
  mixed3 = 3,
  mixed4 = 2,
  conv2d_92 = 1.5
)


# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict <- model$layers
names(layer_dict) <- lapply(layer_dict, function(layer) layer$name) 

# Define the loss.
loss <- k_variable(0) 

for (layer_name in names(layer_contributions)) {
  
  # Add the L2 norm of the features of a layer to the loss.
  coeff <- layer_contributions[[layer_name]]
  activation <- layer_dict[[layer_name]]$output
  scaling <- k_prod(k_cast(k_shape(activation), "float32"))
  loss <- loss + (coeff * k_sum(k_square(activation)) / scaling)
}

names(layer_dict)

# This holds our generated image
dream <- model$input

# Normalize gradients.
grads <- k_gradients(loss, dream)[[1]]
grads <- grads / k_maximum(k_mean(k_abs(grads)), 1e-7)

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs <- list(loss, grads)

fetch_loss_and_grads <- k_function(list(dream), outputs)

eval_loss_and_grads <- function(x) {
  outs <- fetch_loss_and_grads(list(x))
  loss_value <- outs[[1]]
  grad_values <- outs[[2]]
  list(loss_value, grad_values)
}


gradient_ascent <- function(x, iterations, step, max_loss = NULL) {
  s <- dim(x)[2]
  for (i in 1:iterations) {
    c(loss_value, grad_values) %<-% eval_loss_and_grads(x)
    if (!is.null(max_loss) && loss_value > max_loss)
      break
    cat("...Loss value at", i, ":", loss_value, "\n")
    x <- x + (step * grad_values)
    if (i %% 2 == 0) save_img(x, fname=glue("./images/dream/img_{s}_{formatC(i, 3, 3, flag = '0')}.jpg"))
  }
  x
}

resize_img <- function(img, size){
  image_array_resize(img, size[[1]], size[[2]])
}


save_img <- function(img, fname){
  img <- deprocess_image(img)
  image_array_save(img, fname)
}

# Util function to open, resize, and format pictures into appropriate tensors
preprocess_image <- function(image_path) {
  image_load(image_path, target_size = c(500,500)) %>% 
    image_to_array() %>% 
    array_reshape(dim = c(1, dim(.))) %>% 
    inception_v3_preprocess_input()
}

# Util function to convert a tensor into a valid image
deprocess_image <- function(img) {
  img <- array_reshape(img, dim = c(dim(img)[[2]], dim(img)[[3]], 3))
  img <- img / 2
  img <- img + 0.5
  img <- img * 255
  
  dims <- dim(img)
  img <- pmax(0, pmin(img, 255))
  dim(img) <- dims
  img
}

# Playing with these hyperparameters will also allow you to achieve new effects
step <- 0.01          # Gradient ascent step size
num_octave <- 3       # Number of scales at which to run gradient ascent
octave_scale <- 1.4   # Size ratio between scales
iterations <- 60      # Number of ascent steps per scale

# If our loss gets larger than 10,
# we will interrupt the gradient ascent process, to avoid ugly artifacts
max_loss <- 50 

# Fill this to the path to the image you want to use
dir.create("dream")
base_image_path <- "./images/giul_square.jpg"

# Load the image into an array
img <- preprocess_image(base_image_path)

# We prepare a list of shapes
# defining the different scales at which we will run gradient ascent
original_shape <- dim(img)[-1]
successive_shapes <- list(original_shape)
for (i in 1:num_octave) { 
  shape <- as.integer(original_shape / (octave_scale ^ i))
  successive_shapes[[length(successive_shapes) + 1]] <- shape 
}

# Reverse list of shapes, so that they are in increasing order
successive_shapes <- rev(successive_shapes) 

# Resize the array of the image to our smallest scale
original_img <- img 
shrunk_original_img <- resize_img(img, successive_shapes[[1]])

shape <- successive_shapes[[3]]


for (shape in successive_shapes) {
  cat("Processsing image shape", shape, "\n")
  img <- resize_img(img, shape)
  save_img(img, fname=glue("./images/dream/img_{s}_000.jpg", s=dim(img)[2]))
  img <- gradient_ascent(img,
                         iterations = iterations,
                         step = step,
                         max_loss = max_loss)
  upscaled_shrunk_original_img <- resize_img(shrunk_original_img, shape)
  same_size_original <- resize_img(original_img, shape)
  lost_detail <- same_size_original - upscaled_shrunk_original_img
  
  img <- img + lost_detail
  shrunk_original_img <- resize_img(original_img, shape)
  save_img(img, fname = sprintf("./images/dream/at_scale_%s.png",
                                paste(shape, collapse = "x")))
}

save_img(img, fname = "./images/dream/final_dream.png")
plot(as.raster(deprocess_image(img) / 255))