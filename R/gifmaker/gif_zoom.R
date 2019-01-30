library(keras)

img <- image_load("./images/jugarden.jpg", target_size = c(500,500)) %>% 
  image_to_array()

plotImage <- . %>% 
  (function(x) x/255) %>% # normalize
  as.raster() %>% 
  plot(axes=F, box=F)
  
imageZoom <- function(img, zfactor=1.1){
  
  img.size <- dim(img)[1]
  img.new.size <- zfactor*img.size
  #img.center <- dim(zfactor)/2
  img.border.min <- round((img.new.size - img.size)/2)
  img.border.max <- img.border.min + img.size
  
  img %>% 
    image_array_resize(img.new.size, img.new.size) %>% 
    .[img.border.min:img.border.max, img.border.min:img.border.max, ] %>% 
    return()
}


# example 1: simple animated countdown from 10 to "GO!".
jpeg(file="./images/giftest/image_zoom_%03d.jpg", width=500, height=500)
for(i in 1:10){
  plotImage(img)
  img <- imageZoom(img)
}
dev.off()
