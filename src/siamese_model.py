import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import os

def _make_embedding(): 
    inp = Input(shape=(105,105,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = _make_embedding()

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
l1 = L1Dist()

# making a new siamese model
def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# loading existing siamese model
siamese_model = tf.keras.models.load_model( os.path.abspath( '/Users/arjunq21/Documents/python/presence-ai/siamese_modelv2.h5'), 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})





# def resize_image(image_path):   
#     make_sure_image_exists(image_path)
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     print("Resizing image: {}x{}".format(height, width))
#     if height > width:
#         new_height = 1000
#         new_width = int(width * (1000/height))
#     else:
#         new_width = 1000
#         new_height = int(height * (1000/width))
#     resized_image = cv2.resize(image, (new_width, new_height))
#     cv2.imwrite(image_path, resized_image)
#     return 0

# def preprocess(image_path):
#     make_sure_image_exists(image_path)
#     byte_image = tf.io.read_file(image_path)
# #     print(byte_image)
#     img = tf.io.decode_jpeg(byte_image)
# #     print(img)
# #     plt.imshow(img)
#     img = tf.image.resize(img, (105, 105))

#     img = img / 255.0 
# #     plt.imshow(img)
#     #     plt.n
#     return img



# def image_needs_resizing(image_path):
#     make_sure_image_exists(image_path)
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     if height > 1000 or width > 1000:
#         return True
#     return False

# def resize_image(image_path):   
#     make_sure_image_exists(image_path)
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     print("Resizing image: {}x{}".format(height, width))
#     if height > width:
#         new_height = 1000
#         new_width = int(width * (1000/height))
#     else:
#         new_width = 1000
#         new_height = int(height * (1000/width))
#     resized_image = cv2.resize(image, (new_width, new_height))
#     cv2.imwrite(image_path, resized_image)
#     return 0