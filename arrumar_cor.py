#Usa esse código para depois que der resize nas imagens, pois elas ficam com peso misto
from PIL import Image
  
# Import an image from directory:
input_image = Image.open("/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/img_cmp_new_batch8/000000213")
input_image =  input_image.convert("RGB")
  
# Extracting pixel map:
pixel_map = input_image.load()
  
# Extracting the width and height 
# of the image:
width, height = input_image.size

for i in range(width):
    for j in range(height):
        if (pixel_map[i, j][0] >= 110 and pixel_map[i, j][1]>= 110):
        	pixel_map[i, j] = (255,255,0)
        elif (pixel_map[i, j][0] >= 128):
        	pixel_map[i, j] = (255,0,0)
        elif (pixel_map[i, j][1] >= 128):
        	pixel_map[i, j] = (0,255,0)
        else:
        	pixel_map[i,j] = (0,0,0)

#cor=[]
#for i in range(width):
#    for j in range(height):
#        cor.append(pixel_map[i, j])
  
# Saving the final output
input_image.save("000000213ç", format="png")

#print (cor)
#cors=[]
#for i in cor:
#	if i not in cors:
#		cors.append(i)
#print(cors)



# use input_image.show() to see the image on the
# output screen.
#input_image.show()
