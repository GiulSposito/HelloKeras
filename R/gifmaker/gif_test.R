# generate some plots and compose then in a animated gif
if (!file.exists("./images/giftest")) dir.create("./images/giftest")

# example 1: simple animated countdown from 10 to "GO!".
png(file="./images/giftest/example%02d.png", width=200, height=200)
for (i in c(10:1, "G0!")){
  plot.new()
  text(.5, .5, i, cex = 6)
}
dev.off()

# imagemagick call
system("convert -delay 80 ./images/giftest/*.png ./images/giftest/example_1.gif")
file.remove(list.files(path="./images/giftest/",pattern=".png", full.names = T))
