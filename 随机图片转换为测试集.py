# coding: utf-8
#!/usr/bin/python
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os

n=0
list_dirs = os.walk("random_pic")
for root, dirs, files in list_dirs:
    for f in files: 
        n=n+1


data_imgs=np.zeros((n,96,96,3), dtype=np.uint8)
index=0
list_dirs = os.walk("random_pic")
for root, dirs, files in list_dirs:
    for f in files: 
        img=os.path.join(root, f)
        img=Image.open(img)
        com_img=img.resize((96, 96))
        a=np.asarray(com_img)
        data_imgs[index,:,:]=a
        index=index+1


x_test = data_imgs

np.savez('testdata',x_test=x_test)
