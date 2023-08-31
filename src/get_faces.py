import face_recognition as fr
import time
from functions import image_needs_resizing, resize_image

def get_faces_in(image_path):
    if(image_needs_resizing(image_path)):
        resize_image(image_path)

    image = fr.load_image_file(image_path)
    start_time = time.time() 
    face_locations = fr.face_locations(image, number_of_times_to_upsample=1, model='hog')
    print("Found faces: {}, upsampling count: {}, time taken: {} seconds.".format(len(face_locations), 2, (time.time() - start_time)))
    return face_locations

# get_faces_in("test-images/group_pic.jpg") 


