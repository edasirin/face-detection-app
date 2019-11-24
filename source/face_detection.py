import os

import cv2 # OpenCV for image editing, computer vision and deep learning
import numpy as np # Numpy for math/array operations
from source.utils import get_folder_dir # Custom function for better directory name handling


def detect_faces_with_haar_cascade(image, scale_factor = 1.2, min_neighbors = 5):
    '''Detect face in an image'''
    
    faces_list = []

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Detect multiscale images (some images may be closer to camera than others)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # result is a list of faces
    
    # Calculate number of detections
    num_detected_faces = len(detected_faces)
    
    # If not face detected, return empty list  
    if  num_detected_faces == 0:
        return faces_list
    
    for index in range(0, num_detected_faces):
        # Rearrange detection coords
        start_x,start_y,width,height=detected_faces[index]
        end_x=start_x+width
        end_y=start_y+height
        # Initialize a dictionary that will contain detection details
        face_dict = {}
        # Add detection coords to dictionary
        face_dict['rect'] = (start_x, start_y, end_x, end_y)
        # Crop detection region from image and add it to dictionary
        face_dict['face'] = image[start_y : end_y, start_x : end_x]
        # Since there is no detection probablity, assign it empty
        face_dict['prob'] = []
        # Populate detection list
        faces_list.append(face_dict)
        
    # Return the face image area and the face rectangle
    return faces_list
