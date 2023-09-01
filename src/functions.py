import tensorflow as tf
import os
import shutil
import cv2

def make_sure_image_exists(image_path):
    if not os.path.exists(image_path):
        raise Exception("File not found at: {}".format(image_path))
    return 0

# function to convert image to (105, 105)
def preprocess(image_path):
    make_sure_image_exists(image_path)
    byte_image = tf.io.read_file(image_path)
#     print(byte_image)
    img = tf.io.decode_jpeg(byte_image)
#     print(img)
#     plt.imshow(img)
    img = tf.image.resize(img, (105, 105))

    img = img / 255.0 
#     plt.imshow(img)
    #     plt.n
    return img



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




def empty_directory(directory_path):
    # Iterate over all files and subdirectories in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            # Remove files
            os.remove(item_path)
        elif os.path.isdir(item_path):
            # Remove subdirectories and their contents recursively
            shutil.rmtree(item_path)

