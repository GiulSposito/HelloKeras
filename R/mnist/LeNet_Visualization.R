# load lib
library(keras)

# loading and preparing dataset
mnist <- dataset_mnist() 

# separate the datasets
x.train <- mnist$train$x
lbl.train <- mnist$train$y
x.test <-  mnist$test$x
lbl.test <-  mnist$test$y

# let's see what we have
str(x.train)
str(lbl.train)
summary(x.train)


# Redefine dimension of train/test inputs to 2D "tensors" (28x28x1)
x.train <- array_reshape(x.train, c(nrow(x.train), 28,28,1))
x.test  <- array_reshape(x.test,  c(nrow(x.test),  28,28,1))

# normalize values to be between 0.0 - 1.0
x.train <- x.train/255
x.test  <- x.test/255

str(x.train)
summary(x.train)


# one hot encoding
y.train <- to_categorical(lbl.train,10)
y.test  <- to_categorical(lbl.test,10)

str(y.train)


# plot one case
show_digit <- function(tensor, col=gray(12:1/12), ...) {
  tensor %>% 
    apply(., 2, rev) %>%      # reorient to make a 90 cw rotation
    t() %>%                   # reorient to make a 90 cw rotation
    image(col=col, axes=F, asp=1, ...)       # plot matrix as image
}

# check some data
par(mfrow=c(1,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:5) show_digit(x.train[i,,,])
print(lbl.train[1:5])

# build lenet
keras_model_sequential() %>% 
  layer_conv_2d(input_shape=c(28,28,1), filters=20, kernel_size = c(5,5), activation = "tanh") %>% 
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>% 
  layer_conv_2d(filters = 50, kernel_size = c(5,5), activation="tanh" ) %>% 
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2) ) %>% 
  layer_dropout(rate=0.3) %>% 
  layer_flatten() %>% 
  layer_dense(units = 500, activation = "tanh" ) %>% 
  layer_dropout(rate=0.3) %>% 
  layer_dense(units=10, activation = "softmax") -> model

# lets look the summary
summary(model)

# keras compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# training
system.time(
  history <- model %>% fit(
    x.train, y.train, epochs=30, batch_size=128,
    validation_split=0.3
  )
)

# evaluating the model
evaluate(model, x.test, y.test)

# save/load the model
save_model_hdf5(model, "./models/mnist_lenet.hdf5")
saveRDS(history, "./models/mnist_lenet_history.rds")
model <-  load_model_hdf5("./models/mnist_lenet.hdf5")

# --- visualize the activation patterns ---------------------------------------------

# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)

# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# choose a case
digit_tensor <- array_reshape(x.train[45,,,], c(1,28,28,1))

# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(digit_tensor)

# plot a tensor channel
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(channel[,nrow(channel):1], axes = FALSE, asp = 1)
}

# plot the channels of a layout ouput (activation)
plotActivations <- function(.activations, .index){
  layer_inpected <- .activations[[.index]]
  par(mfrow=c(dim(layer_inpected)[4]/5,5), mar=c(0.1,0.1,0.1,0.1))
  for(i in 1:dim(layer_inpected)[4]) plot_channel(layer_inpected[1,,,i])
}

# look the 2D layers activations
plotActivations(activations, 1) # conv2D - tanh
plotActivations(activations, 2) # max pooling
plotActivations(activations, 3) # conv2D - tanh
plotActivations(activations, 4) # max pooling

# --- visualization the filters learned patterns -----------------------------------

# clean graph output settings
par(mfrow=c(1,1))

# generating a noise baseline
img <- array(runif(28 * 28 * 1), dim = c(1, 28, 28, 1))

img[1,,,1] %>%  # unique channel
  image(axes=F, asp=1)

# define parameters for the gradient ascent
k_set_learning_phase(0)

# This will contain our generated image
dream <- model$input

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict <- model$layers
names(layer_dict) <- map_chr(layer_dict ,~.x$name)

# Define the loss
loss <- k_variable(0.0)

# Add the L2 norm of the features of a layer to the loss
layer_name <- "conv2d_1"
coeff <- 0.05
x <- layer_dict[[layer_name]]$output
scaling <- k_prod(k_cast(k_shape(x), 'float32'))

# Avoid border artifacts by only involving non-border pixels in the loss
loss <- loss + coeff*k_sum(k_square(x)) / scaling

# Compute the gradients of the dream wrt the loss
grads <- k_gradients(loss, dream)[[1]] 

# Normalize gradients.
grads <- grads / k_maximum(k_mean(k_abs(grads)), k_epsilon()) # k_epsilon to avoid 0 division

# Set up function to retrieve the value
# of the loss and gradients given an input image.
fetch_loss_and_grads <- k_function(list(dream), list(loss,grads))

eval_loss_and_grads <- function(image){
  outs <- fetch_loss_and_grads(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = outs[[2]]
  )
}

# do the gradient ascent
gradient_ascent <- function(x, iterations, step, max_loss = NULL) {
  for (i in 1:iterations) {
    out <- eval_loss_and_grads(x)
    if (!is.null(max_loss) & out$loss_value > max_loss) {
      break
    } 
    print(paste("Loss value at", i, ':', out$loss_value))
    x <- x + step * out$grad_values
  } 
  x
}

# Playing with these hyperparameters will also allow you to achieve new effects
step <- 0.02  # Gradient ascent step size
num_octave <- 3  # Number of scales at which to run gradient ascent
octave_scale <- 1.4  # Size ratio between scales
iterations <- 1000  # Number of ascent steps per scale
max_loss <- 10

img.resp <- gradient_ascent(img, iterations, step, max_loss)

img.resp[1,,,1] %>% 
  image(axes=F, asp=1)
