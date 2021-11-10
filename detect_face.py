import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

def verify_image(filename):
    im = Image.load(filename)
    im.verify() 
    im.close()
  

input = './drag3/0.jpg'

padding = 25

def detect_faces(filename, haarfile='haarcascade_frontalface_alt2.xml', scaleFactor=1.25, minNeighbors=3, minSize=(40,40)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haarfile)
    
    img = cv2.imread(filename)
    # Make a copy of the original crop with
    image_copy = np.copy(filename)

    # Convert the image to gray 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # return array of x,y,w,h
    return face_cascade.detectMultiScale(gray_image, scaleFactor, minNeighbors, minSize)

def crop_faces(original, faces):
    crops = []

    # Draw rectangle around the faces
    for count,face in enumerate(faces):
        x, y, w, h = face
        
        left = x - padding
        right = x + w + padding
        top = y - padding
        bottom = y + h + padding
        crop = original[left:right,top:bottom]

        crops.append({'img':crop,'weight':x+y})
        
    # ensure that faces are sorted in the order in which they appear
    crops.sort(key=lambda x: x['weight'])

    return [crop['img'] for crop in crops]

def crop_and_save_faces(original, faces, output_base_path, output_size=(224,224)):
    crops = crop_faces(original, faces)

    # create directory if doesn't exist
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    for i, im in enumerate(crops):
        resized = cv2.resize(im, output_size, interpolation = cv2.INTER_AREA)
        filename = f'{output_base_path}/{i+1}.jpg'
        cv2.imwrite(filename, resized)
    

