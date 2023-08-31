import tensorflow as tf
from functions import preprocess, make_sure_image_exists
from siamese_model import siamese_model

# data = [
#     ("all-pics/Angelina_Jolie_0010.jpg", "all-pics/Angelina_Jolie_0011.jpg", 1),
#     ("all-pics/Aaron_Tippin_0001.jpg", "all-pics/Abba_Eban_0001.jpg", 0),
# ]


def predict(anchor_image, validation_image):
    map(lambda x: make_sure_image_exists(x), [anchor_image, validation_image])
    data = tf.data.Dataset.zip(
        tf.data.Dataset.from_tensor_slices([preprocess(anchor_image)]),
        tf.data.Dataset.from_tensor_slices([preprocess(validation_image)]),
    )
    data = data.batch(1)
    test_inp, test_val = data.as_numpy_iterator().next()
    predictions = siamese_model.predict([test_inp, test_val])
    return predictions[0]

# print(predict(data[1][0], data[1][1]))