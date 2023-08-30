import face_recognition as fr
import time
import os
import cv2

def make_sure_image_exists(image_path):
    if not os.path.exists(image_path):
        raise Exception("File not found at: {}".format(image_path))
    return 0

def image_needs_resizing(image_path):
    make_sure_image_exists(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height > 1000 or width > 1000:
        return True
    return False

def resize_image(image_path):   
    make_sure_image_exists(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    print("Resizing image: {}x{}".format(height, width))
    if height > width:
        new_height = 1000
        new_width = int(width * (1000/height))
    else:
        new_width = 1000
        new_height = int(height * (1000/width))
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(image_path, resized_image)
    return 0

def get_faces_in(image_path):
    if(image_needs_resizing(image_path)):
        resize_image(image_path)

    image = fr.load_image_file(image_path)
    start_time = time.time() 
    face_locations = fr.face_locations(image, number_of_times_to_upsample=2, model='cnn')
    print("Found faces: {}, upsampling count: {}, time taken: {} seconds.".format(len(face_locations), 2, (time.time() - start_time)))
    return face_locations

# get_faces_in("test-images/group_pic.jpg") 


