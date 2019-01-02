# diretorio base do download
basedir <- "~gsposito/downloads/dogs-vs-cats"
dir(basedir) # check

sourcedir <- file.path(basedir, "train.data")
dir(sourcedir)

# diretorio de treinamento
train.dir <- file.path(basedir, "train")
dir.create(train.dir) # check

# diretorio de testes
test.dir <- file.path(basedir, "test")
dir.create(test.dir) #

#cria um diretorio para validacao
valid.dir <- file.path(basedir, "valid")
dir.create(valid.dir)

# separa dogs e cats nos diretorios de trainamento, validacao e testes
train_cats_dir <- file.path(train.dir, "cats")
dir.create(train_cats_dir)

train_dogs_dir <- file.path(train.dir, "dogs")
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(valid.dir, "cats")
dir.create(validation_cats_dir)

validation_dogs_dir <- file.path(valid.dir, "dogs")
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test.dir, "cats")
dir.create(test_cats_dir)

test_dogs_dir <- file.path(test.dir, "dogs")
dir.create(test_dogs_dir)

# separa arquivos nos respectivos diretorios
fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(train_cats_dir))

fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(validation_cats_dir))

fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(train_dogs_dir))

fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(validation_dogs_dir))

fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(sourcedir, fnames),
          file.path(test_dogs_dir))

# sanity check
cat("total training cat images:", length(list.files(train_cats_dir)), "\n")
cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")
cat("total validation dog images:", length(list.files(validation_dogs_dir)), "\n")
cat("total test cat images:", length(list.files(test_cats_dir)), "\n")
cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")
