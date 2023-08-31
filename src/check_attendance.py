# import random 
# def check_attendance(group_image_path, student_images):
#     shuffled_array = random.sample(student_images, len(student_images))
#     return shuffled_array[0 : (len(student_images)//2) ]

# # print(check_attendance("files/group.jpg", ['files/1.jpg', 'files/2.jpg', 'files/4.jpg', 'files/5.jpg', 'files/6.jpg', 'files/7.jpg']))
import uuid
from get_faces import get_faces_in
from predict import predict
from functions import make_sure_image_exists
import cv2

def face_location_to_cropped_image(face_location, cv2img):
    cropped_region =  cv2img[face_location[0]:face_location[2], face_location[3]:face_location[1]]
    print("Cropped region: {}, shape: {}".format(cropped_region, cropped_region.shape))
    # cv2.imshow("Cropped Region: ", cropped_region) 
    return cropped_region

# def check_attendance(group_image_path, student_images):
#     map(lambda x: make_sure_image_exists(x), [group_image_path, *student_images])
#     print("Getting face coordinates")
#     face_coordinates = get_faces_in(group_image_path)
#     print("obtained face coordinates: {}".format(face_coordinates))
#     cv2img = cv2.imread(group_image_path)


def get_individual_faces_in(image_path):
    face_coordinates = get_faces_in(image_path)
    cv2img = cv2.imread(image_path)
    print("image shape: {}".format(cv2img.shape))
    for coordinate in face_coordinates:
        print("Face Coordinates: {}".format(coordinate))
        portion = cv2img[coordinate[0]:coordinate[2], coordinate[3]:coordinate[1]]
        cv2.imwrite("temp-faces/" + str(uuid.uuid4()) + ".jpg", portion)

    # cropped_images = map(lambda x: face_location_to_cropped_image(x, cv2img), face_coordinates)
    # map(lambda x: cv2.imwrite("test-images/" + str(uuid.uuid4()) + ".jpg", x), cropped_images)
    # return cropped_images


get_individual_faces_in("test-images/lumre.jpg")

# img = cv2.imread("test-images/tree_plantation.jpg")

# cropped = img[95:147, 481:533]

# cv2.imwrite("test-images/cropped.jpg", cropped)

# print(str(uuid.uuid4()))