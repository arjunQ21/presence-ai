from detection.get_faces import get_faces_in 
import cv2

image_path = "test-images/group_pic.jpg"

faces = get_faces_in(image_path) 

print(faces)