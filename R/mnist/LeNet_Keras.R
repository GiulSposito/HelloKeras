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
for(i in 1:5) show_digit(x.train[i,])
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
    x.train, y.train, epochs=3, batch_size=128,
    validation_split=0.2
  )
)
