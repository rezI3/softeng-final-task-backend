import os
import subprocess
from PIL import Image
import cv2 as cv

dir0 = 'org'
dir1 = 'png'
dir2 = 'png_resize'
dir3 = 'face'

files0 = os.listdir(dir0)
files0.sort()

for file in files0:

    if '.jpg'  in file:
        command = 'sips --setProperty format png ' + dir0 +'/' + file +  ' --out ' + dir1 +'/' +  file.replace('.jpg','.png')
        subprocess.call(command, shell=True)
        print(file)

files1 = os.listdir(dir1)
files1.sort()

# aaa.jpg

for file in files1:
    if '.png' in file:
        img0 = os.path.join(dir1, file)
        img0_img = Image.open(img0)
        h = img0_img.height
        w = img0_img.width
        img1_img = img0_img.resize((600,round(600*h/w)))
        img1 = os.path.join(dir2, file)
        img1_img.save(img1)
        print(file)

# aaa.png

files2 = os.listdir(dir2)
files2.sort()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv.CascadeClassifier('haarcascade_profileface.xml')

for file in files2:
    if '.png' in file:
        dirfile = os.path.join(dir2, file)
        img = cv.imread(dirfile)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = img[y-10:y+h+10, x-10:x+w+10]
            face_name = str(file.strip('.png'))+'_'+str(x)+'_'+str(y)+'.png'
            dirface = os.path.join(dir3,face_name)
            facefile = cv.imwrite(dirface, face)
            #cv.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
            print(face_name)

# aaa_152_22.png
# aaa_11_70.png
# aaa_438_41.png
# aaa_79_106.png
# aaa_385_140.png
# aaa_190_175.png
# aaa_269_171.png
# aaa_76_206.png
# aaa_527_257.png
# aaa_91_277.png
# aaa_254_330.png
# aaa_446_348.png