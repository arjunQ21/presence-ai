import uuid
from get_faces import get_faces_in
from predict import predict
from functions import make_sure_image_exists, empty_directory
import os
import cv2

# Gets individual faces and stores them in temp-faces folder
def get_individual_faces_in(image_path):
    face_coordinates = get_faces_in(image_path)
    cv2img = cv2.imread(image_path)
    print("image shape: {}".format(cv2img.shape))
    for coordinate in face_coordinates:
        print("Face Coordinates: {}".format(coordinate))
        portion = cv2img[coordinate[0] : coordinate[2], coordinate[3] : coordinate[1]]
        cv2.imwrite(
            os.path.join(os.path.abspath("/Users/arjunq21/Documents/python/presence-ai/temp-faces/"), str(uuid.uuid4()) + ".jpg"),
            portion,
        )

# Parameters:
# `group_image_path` is path of image of users in group, sent from POST Request from Flutter app, when taking attendance.

# `student_images` is an array of file paths of images of students in the group for which attendance is being taken.

# Return Value:

# array of file paths of images of students detected in the provided group_image_path parameter. This array is subset of student_images array.


def check_attendance(group_image_path, student_images):
    for path in [group_image_path, *student_images]:
        make_sure_image_exists(path)

    print("Group image path: {}, student images: {}".format(group_image_path, student_images))

    get_individual_faces_in(group_image_path)

    # print("Getting face coordinates")
    faces = list(
        map(
            lambda x: os.path.join(os.path.abspath("/Users/arjunq21/Documents/python/presence-ai/temp-faces/"), x),
            os.listdir(os.path.abspath("/Users/arjunq21/Documents/python/presence-ai/temp-faces")),
        )
    )

    print(faces)

    def predict_face_presence(single_student_image):
        for face in faces:
            if predict(single_student_image, face) > 0.7:
                return True
        return False

    present_faces = list(map(lambda x: (predict_face_presence(x), x), student_images))
    print("Emptying temp directory")
    empty_directory(os.path.abspath("/Users/arjunq21/Documents/python/presence-ai/temp-faces"))
    return list(map(lambda x: x[1], list(filter(lambda x: x[0], present_faces))))





# get_individual_faces_in("test-images/lumre.jpg")

# print(
#     check_attendance(
#         "test-images/lumre.jpg",
#         [
#             "temp-faces/3d64dc6a-a58c-4afa-a7ac-5828817501f2.jpg",
#         ],
#     )
# )
