# Dogs vs Cats - Direct Case

# load lib
library(keras)

# convnet with dropout
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
  
  layer_dropout(rate = 0.5) %>%  # <= dropout layer kills 50% of synapsis
  
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
# original
# datagen <- image_data_generator(rescale = 1/255)

# data augmentation

# the network will never see the same input twice. 
# But the inputs it sees are still heavily intercorrelated,
# because they come from a small number of original images
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,      # rotation
  width_shift_range = 0.2,  # translations
  height_shift_range = 0.2, # translations
  shear_range = 0.2,        # cuts in the image
  zoom_range = 0.2,         # zooms
  horizontal_flip = TRUE,   # mirrors
  fill_mode = "nearest"
)

# let's see some images
fnames <- list.files(file.path(train.dir,"cats"), full.names = T) # look for images
img_path <- fnames[[round(runif(1,1,length(fnames)))]] # get one random image filename
img <- image_load(img_path, target_size = c(150,150))  # load it
img_array <- image_to_array(img) # tensor it to 150x150x3
img_array <- array_reshape(img_array, c(1,150,150,3)) # to 1x150x150x3

# creates a image generator from data (not from filedir)
augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)

# plot the images
op <- par(mfrow=c(2,2), pty="s", mar=c(1,0,.1,0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)

# Generates batches of data from images in a directory 
train_generator <- flow_images_from_directory(
  train.dir,
  datagen,
  target_size = c(150,150),
  batch_size = 32,  # larger batch, it isn't limited by dataset now
  class_mode = "binary"
)

# Generates batches of data from images in a directory 
valid_generator <- flow_images_from_directory(
  valid.dir,
  datagen,
  target_size = c(150,150),
  batch_size = 32, # larger batch, it isn't limited by dataset now
  class_mode = "binary"
)

# training the model
# fit_generator is equivalente for fit for data generators

system.time(
  history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100, # more trainings
    epochs = 100,          # more epochs
    validation_data = valid_generator,
    validation_steps = 50
  )
)

# save model and history
model %>% save_model_hdf5("./models/cats_and_dogs_small_2.h5")
history %>% saveRDS("./models/cats_and_dogs_small_2_hist.rds")

# check for overvitting (validationd keeps at 70% accuracy from 2 or 3 epocas)
plot(history)

# training results
# loss: 0.3285
# acc:  0.8550
# val_loss: 0.4496  
# val_acc:  0.8144

# Generates batches of data from images in a directory 
test_generator <- flow_images_from_directory(
  test.dir,
  image_data_generator(rescale = 1/255), # test set don't need be augmented!
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

# eval on test data
model %>% 
  evaluate_generator(test_generator, 1000/20) # 1000 imagens em batch of 20

# test results
# loss: 0.4201486
# acc:  0.822
