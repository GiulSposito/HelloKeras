# loading the pretrained model already packaged with keras package

# setup
library(keras)

# VGG16 only Convolutional Layers
conv_base <- application_vgg16(
  weights = "imagenet",      # weights checkpoint
  include_top = F,           # the model don't import classifying layers
  input_shape = c(150,150,3) # define the input shape
)

# check model
summary(conv_base)

# last layer outputs c(4,4,512)
# two strategies:
#  1) run the convbase over the dataset to generate features to be used with other model
#  2) extend the model with new classifying layers

# this script: === strategy 1 ===

# data directories
basedir <- "~gsposito/downloads/dogs-vs-cats"
sourcedir <- file.path(basedir, "train.data")
train.dir <- file.path(basedir, "train")
test.dir  <- file.path(basedir, "test")
valid.dir <- file.path(basedir, "valid")

# data generator
datagen    <- image_data_generator(rescale = 1/255)
batch_size <- 20

# this function runs a prediction on a dataset (directory)
# and reply the last layer's output (features)
extract_features <- function(directory, sample_count) {
  
  # "result vars"
  features <- array(0, dim=c(sample_count, 4, 4, 512)) 
  labels   <- array(0, dim=c(sample_count))
  
  # image flow
  generator <- flow_images_from_directory(
    directory   = directory,
    generator   = datagen, 
    target_size = c(150,150),
    batch_size  = batch_size,
    class_mode  = "binary"
  )

  
  i <- 0
  while(T){ # actualy while "has image" to predict
    
    batch <- generator_next(generator) # get a batch
    inputs_batch <- batch[[1]]         # get inputs
    labels_batch <- batch[[2]]         # get labels
    
    # generate the last layer's output of the batch
    features_batch <- conv_base %>% predict(inputs_batch)
    
    # story then in the "results" vars
    index_range <- ((i*batch_size)+1):((i+1)*batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range]      <- labels_batch
    
    # update the index
    i <- i+1
    if(i*batch_size >= sample_count) break  # check if we need to stop
    
  }
  
  # results
  list(
    features = features,
    labels   = labels
  )
}

# extract the features
train      <- extract_features(train.dir,2000)
validation <- extract_features(valid.dir,1000)
test       <- extract_features(test.dir, 1000)

# classifier model
class_model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(4,4,512)) %>% 
  layer_dense( units=256, activation="relu" ) %>% 
  layer_dropout( rate=0.5 ) %>% 
  layer_dense( units=1, activation = "sigmoid" )

summary(class_model)

class_model %>% 
  compile(
    optimizer = optimizer_rmsprop(lr=2e-5),
    loss      = "binary_crossentropy",
    metrics   = c("accuracy")
  )

# train new classifier
system.time(
  history <- class_model %>% 
    fit(
      train$features, train$labels,
      epochs     = 30,
      batch_size = 20,
      validation_data = list(validation$features, validation$labels)
    )
)

# training stats
# loss: 0.0882
# acc: 0.9725
# val_loss: 0.240
# val_acc: 0.9070 (!!)

# save model and history
class_model %>% save_model_hdf5("./models/cats_and_dogs_preTrained_classifier_1.h5")
history %>% saveRDS("./models/cats_and_dogs_preTrained_classifier_1_hist.h5")

plot(history)

# check in the test set
class_model %>% evaluate(test$features, test$labels)

# test stats
# loss: 0.25572
# acc : 0.886 (!!)


# ===== Strategy 2 - Extending the preTrained ConvNet ====

# new model
model <- keras_model_sequential() %>% 
  conv_base() %>%  # conv layers
  layer_flatten() %>%  # new classifier layers 
  layer_dense( units=256, activation="relu" ) %>% 
  # layer_dropout( rate=0.5 ) %>% 
  layer_dense( units=1, activation = "sigmoid" )

# check "trainable parameters 
summary(model)

# freezing the conv layers to train only the new classifier
cat("This is the number of trainable weights before freezing the conv base:",
    length(model$trainable_weights), "\n")

freeze_weights(conv_base)

cat("This is the number of trainable weights after freezing the conv base:",
    length(model$trainable_weights), "\n")

# check trainable parameters
summary(conv_base)

# compile
model %>% 
  compile(
    optimizer = optimizer_rmsprop(lr=2e-5),
    loss      = "binary_crossentropy",
    metrics   = c("accuracy")
  )

# image generator with data augmentation
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# the test hasn't data augmentation (sure!)
test_datagen <- image_data_generator(rescale = 1/255)

# train and validation "image flows"
train_generator <- flow_images_from_directory(
  train.dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  valid.dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# train extende model
system.time(
  history <- model %>% 
    # fit_generator  run in parallel to the model, for efficiency. For instance, 
    # this allows you to do real-time data augmentation on images on CPU in parallel 
    # to training your model on GPU.
    fit_generator(
      train_generator,
      steps_per_epoch = 100,
      epochs = 30,
      validation_data = validation_generator,
      validation_steps = 50
    )
)

plot(history)

class_model %>% save_model_hdf5("./models/cats_and_dogs_preTrained_extended_2.h5")
history %>% saveRDS("./models/cats_and_dogs_preTrained__extended_2_hist.h5")

# train stat
# loss: 0.2775
# acc: 0.8875
# val_loss: 0.2376
# val_acc: 0.9040

# eval the new model

# test generator
test_generator <- flow_images_from_directory(
  test.dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# eval
model %>% evaluate_generator(test_generator, 1000/20)

# eval stat
# loss: 0.2423
# acc:  0.902

# ================= fine tunning =====================
# 1. Add your custom network on top of an already-trained base network.
# 2. Freeze the base network.
# 3. Train the part you added.
# 4. Unfreeze some layers in the base network.
# 5. Jointly train both these layers and the part you added.

# from step 4....

summary(conv_base)
unfreeze_weights(conv_base, from = "block5_conv1")
summary(conv_base)

# compile
model %>% 
  compile(
    optimizer = optimizer_rmsprop(lr=2e-5),
    loss      = "binary_crossentropy",
    metrics   = c("accuracy")
  )

# train extende model
system.time(
  history <- model %>% 
    # fit_generator  run in parallel to the model, for efficiency. For instance, 
    # this allows you to do real-time data augmentation on images on CPU in parallel 
    # to training your model on GPU.
    fit_generator(
      train_generator,
      steps_per_epoch = 100,
      epochs = 100,
      validation_data = validation_generator,
      validation_steps = 50
    )
)

# train stats
# loss:     0.0177 
# acc:      0.9945 
# val_loss: 0.3358 
# val_acc:  0.9440

plot(history)

model %>% save_model_hdf5("./models/cats_and_dogs_preTrained_fineTunned_3.h5")
history %>% saveRDS("./models/cats_and_dogs_preTrained_fineTunned_3_hist.h5")

# eval
model %>% evaluate_generator(test_generator, 1000/20)

# test stats
# loss: 0.4248
# acc:  0.94