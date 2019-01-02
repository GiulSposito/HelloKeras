library(keras)

# loading and preparing dataset
mnist <- dataset_mnist()

x.train <- mnist$train$x
lbl.train <- mnist$train$y
x.test <-  mnist$test$x
lbl.test <-  mnist$test$y

dim(x.train) <- c(nrow(x.train), 784)
dim(x.test ) <- c(nrow(x.test ), 784)

x.train <- x.train/255
x.test  <- x.test/255

y.train <- to_categorical(lbl.train,10)
y.test  <- to_categorical(lbl.test,10)


# plot one case
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  matrix(arr784, nrow=28) %>% # convert to matrix 28 x 28
    apply(., 2, rev) %>%      # reorient to make a 90 cw rotation
    t() %>%                   # reorient to make a 90 cw rotation
    image(col=col, axes=F, asp=1, ...)       # plot matrix as image
}


# check some data
par(mfrow=c(1,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:5) show_digit(x.train[i,,,])
lbl.train[1:5]

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

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Redefine  dimension of train/test inputs
x.train <- array_reshape(x.train, c(nrow(x.train), 28,28,1))
x.test  <- array_reshape(x.test,  c(nrow(x.test),  28,28,1))

system.time(
  history <- model %>% fit(
    x.train, y.train, epochs=1, batch_size=128,
    validation_split=0.3
  )
)

save_model_hdf5(model, "./models/mnist_conv_tanh_1epoch.hdf5")


# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)

# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

digit_tensor <- array_reshape(x.train[45,,,], c(1,28,28,1))

# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(digit_tensor)

first_layer_activations <- activations[[3]]
dim(first_layer_activations)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(channel[,nrow(channel):1], axes = FALSE, asp = 1)
}

par(mfrow=c(10,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:50) plot_channel(first_layer_activations[1,,,i])


evaluate(model, x.test, y.test)
