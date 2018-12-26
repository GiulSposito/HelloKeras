library(keras)

mnist <- dataset_mnist()

x.train <- mnist$train$x
y.train <- mnist$train$y
x.test <-  mnist$test$x
y.test <-  mnist$test$y

dim(x.train) <- c(nrow(x.train), 784)
dim(x.test ) <- c(nrow(x.test ), 784)

x.train <- x.train/255
x.test  <- x.test/255


y.train <- to_categorical(y.train,10)
y.test  <- to_categorical(y.test,10)

head(y.train)
head(x.train)


model <- keras_model_sequential()

model %>% 
  layer_dense(units=256, activation="relu", input_shape = c(784)) %>% 
  layer_dropout(rate=0.4) %>% 
  layer_dense(units=128, activation = "relu") %>% 
  layer_dropout(rate=0.3) %>% 
  layer_dense(units=10, activation = "softmax")

model %>% summary()

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = "accuracy"
)

system.time(
  history <- model %>% fit(
    x.train, y.train, epochs=30, batch_size=128,
    validation_split=0.2
  )
)


# layers.outputs <- lapply(model$layers[1:5], function(layer) layer$output)
# 
# activation.model <- keras_model(inputs = model$input, outputs = layers.outputs)
# 
# activations <- activation.model %>% 
#   predict(x.test)
# 
# fl.activation <- activations[[1]]
# dim(fl.activation)
# 
# array_reshape(fl.activation[3,], c(16,16)) %>% 
#   image(axes=F)

library(caret)

model %>% 
  predict_classes(x.test) %>% 
  as.integer() %>% 
  as.factor() %>% 
  confusionMatrix(reference=as.factor(as.vector(mnist$test$y, mode="integer")))

