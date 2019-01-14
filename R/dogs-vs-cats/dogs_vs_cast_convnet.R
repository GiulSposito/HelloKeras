library(keras)

# convnet 
model <- keras_model_sequential() %>% 
  layer_conv_2d( filter=32, kernel_size = c(3,3), activation = "relu", input_shape = c(150,150,3) ) %>% 
  layer_max_pooling_2d( pool_size = c(2,2)) %>% 
  
  layer_conv_2d( filters=64, kernel_size = c(3,3), activation = "relu" ) %>% 
  layer_max_pooling_2d( pool_size = c(2,2) ) %>% 
  
  layer_conv_2d( filters=128, kernel_size=c(3,3), activation="relu" ) %>% 
  layer_max_pooling_2d( pool_size = c(2,2)) %>% 
  
  layer_conv_2d( filters=128, kernel_size=c(3,3), activation="relu" ) %>% 
  layer_max_pooling_2d( pool_size = c(2,2)) %>% 

  layer_flatten() %>% 
  layer_dense( units=512, activation="relu" ) %>% 
  layer_dense( units=1, activation = "sigmoid" )

summary(model)

# compilation
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("acc")
)

# 1) Read the picture files.
# 2) Decode the JPEG content to RGB grids of pixels.
# 3) Convert these into floating-point tensors.
# 4) Rescale the pixel values (between 0 and 255) to the [0, 1] interval

# data directories
basedir <- "~gsposito/downloads/dogs-vs-cats"
sourcedir <- file.path(basedir, "train.data")
train.dir <- file.path(basedir, "train")
test.dir <- file.path(basedir, "test")
valid.dir <- file.path(basedir, "valid")


# automatically turn image files on disk into batches of pre-processed tensors.
datagen <- image_data_generator(rescale = 1/255)

# Generates batches of data from images in a directory 
train_generator <- flow_images_from_directory(
  train.dir,
  datagen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

# Generates batches of data from images in a directory 
valid_generator <- flow_images_from_directory(
  valid.dir,
  datagen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

# see how the batch works
# Let’s look at the output of one of these generators: it yields batches of 150 × 150
# RGB images (shape 20,150,150,3) and binary labels (shape 20).

batch <- generator_next(train_generator) # get one training batch 20 images of 150w x 150h x 3colors
str(batch)

# training the model
# fit_generator is equivalente for fit for data generators

system.time(
  history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = valid_generator,
    validation_steps = 50
  )
)

# save model and history
model %>% save_model_hdf5("./models/cats_and_dogs_small_1.h5")
history %>% saveRDS("./models/cats_and_dogs_small_1_hist.rds")

# check for overvitting (validationd keeps at 70% accuracy from 2 or 3 epocas)
plot(history)


