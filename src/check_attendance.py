import uuid
from get_faces import get_faces_in
from predict import predict
from functions import make_sure_image_exists, empty_directory
import os
import cv2


def check_attendance(group_image_path, student_images):
    for path in [group_image_path, *student_images]:
        make_sure_image_exists(path)
    # print("Getting face coordinates")
    faces = list(map(lambda x: "temp-faces/" + x, os.listdir("temp-faces")))
    def predict_face_presence(single_student_image):
        for face in faces:
            if predict(single_student_image, face) > 0.7:
                return True
        return False
    present_faces = list(map(lambda x: (predict_face_presence(x), x), student_images))
    print("Emptying temp directory")
    empty_directory("temp-faces")
    return list(map(lambda x: x[1], list(filter(lambda x: x[0], present_faces))))


# Gets individual faces and stores them in temp-faces folder
def get_individual_faces_in(image_path):
    face_coordinates = get_faces_in(image_path)
    cv2img = cv2.imread(image_path)
    print("image shape: {}".format(cv2img.shape))
    for coordinate in face_coordinates:
        print("Face Coordinates: {}".format(coordinate))
        portion = cv2img[coordinate[0] : coordinate[2], coordinate[3] : coordinate[1]]
        cv2.imwrite("temp-faces/" + str(uuid.uuid4()) + ".jpg", portion)


# get_individual_faces_in("test-images/lumre.jpg")

print(
    check_attendance(
        "test-images/lumre.jpg",
        [
            "temp-faces/08b44451-c5c4-4652-8af7-dc2f96f2b8a8.jpg",
        ],
    )
)
