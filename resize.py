from skimage import transform, data, io
from PIL import Image
import os
import cv2

infile = "/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/img_cmp_new_batch8/000000213.png"
outfile = '/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/000000213.png'

''' PIL'''
def fixed_size1(width, height):
    im = Image.open(infile)
    out = im.resize((width, height),Image.ANTIALIAS)
    out.save(outfile)

''' open cv'''
def fixed_size2(width, height):
    img_array = cv2.imread(infile)
    new_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outfile, new_array)


def fixed_size3(width, height):
    img = io.imread(infile)
    dst = transform.resize(img, (512, 512))
    io.imsave(outfile, dst)

fixed_size2(512, 512)
