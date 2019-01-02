library(keras)

imdb <-  dataset_imdb(num_words=10000)
train.x <- imdb$train$x
train.y <- imdb$train$y %>% as.numeric()
test.x <- imdb$test$x
test.y <- imdb$test$y %>% as.numeric()

word.idx <- dataset_imdb_word_index()
str(word.idx)


vec.sequences <- function(sequences, dimension=10000){
  
  results <- matrix(0, nrow=length(sequences), ncol=dimension)
  
  for(i in 1:length(sequences)){
    results[i, sequences[[i]]] <- i
  }
  
  return(results)
}

train.x <- vec.sequences(train.x) 
test.x  <- vec.sequences(test.x)

keras_model_sequential() %>% 
  layer_dense(units = 100, activation = "tanh", input_shape=c(10000)) %>% 
  layer_dense(units = 32, activation = "tanh") %>% 
  layer_dense(units = 16, activation = "tanh") %>% 
  layer_dense(units = 10, activation = "tanh") %>% 
  layer_dense(units = 01, activation = "sigmoid") -> model

model %>% 
  compile(
    optimizer = "rmsprop",
    loss      = "binary_crossentropy",
    metrics   = c("accuracy")
  )

history <- model %>% fit(
  train.x, train.y,
  epochs     =  20, 
  batch_size = 512,
  validation_split = .35
)

model %>% evaluate(test.x, test.y)
