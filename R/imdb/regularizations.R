library(keras)

# load database
imdb <-  dataset_imdb(num_words=10000)
train.x <- imdb$train$x
train.y <- imdb$train$y %>% as.numeric()
test.x <- imdb$test$x
test.y <- imdb$test$y %>% as.numeric()

rm(imdb)
gc()

# vectorize "one hot encode"
vec.sequences <- function(sequences, dimension=10000){
  
  results <- matrix(0, nrow=length(sequences), ncol=dimension)
  
  for(i in 1:length(sequences)){
    results[i, sequences[[i]]] <- i
  }
  
  return(results)
}

train.x <- vec.sequences(train.x) 
test.x  <- vec.sequences(test.x)

gc()

# original network
org.model <- keras_model_sequential() %>% 
  layer_dense( units=63, activation="tanh", input_shape = c(10000) ) %>% 
  layer_dense( units=32, activation="tanh", input_shape = c(10000) ) %>% 
  layer_dense( units=16, activation = "tanh" ) %>% 
  layer_dense( units=8, activation = "tanh" ) %>% 
  layer_dense( units=1,  activation = "sigmoid" )

org.model %>% 
  compile(
    optimizer = "rmsprop",
    loss      = "binary_crossentropy",
    metrics   = c("accuracy")
  )

val.seq <- sample(1:25000, 10000)

history <- org.model %>% fit(
  train.x[-val.seq,], train.y[-val.seq],
  epochs     =  20, 
  batch_size = 512,
  validation_data = list(train.x[val.seq,], train.y[val.seq])
)

org.model %>% evaluate(test.x, test.y)

