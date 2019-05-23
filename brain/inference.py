from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions
)
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes')
parser.add_argument('image')
parser.add_argument('--top_n', type=int, default=10)


def predict(img_path):
    # create model
    model = load_model('../DogBreedPredictor/model_fine_final.h5')
    model._make_predict_function()

    # load class names
    with open('../DogBreedPredictor/brain/classes.txt', 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

    # load an input image
    orig = np.array(cv2.imread(img_path))

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda y: y[1])
    (class_name, prob) = result[0]
    cv2.putText(orig, "Class name:", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 155), 2)
    cv2.putText(orig, class_name[10:], (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 155), 2)
    cv2.putText(orig, "Probability: ", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 155), 2)
    cv2.putText(orig, "{:.2f}%".format(prob), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 155), 2)

    # for i in range(2):
    #     (class_name, prob) = result[i]
    #     cv2.putText(x, "Top %d ====================" % (i + 1), (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.2, (43, 99, 255), 2)
    #     cv2.putText(x, "Class name: %s" % class_name, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.2, (43, 99, 255), 2)
    #     cv2.putText(x, "Probability: %.2f%%" % prob, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.2, (43, 99, 255), 2)

        # print("Top %d ====================" % (i + 1))
        # print("Class name: %s" % class_name)
        # print("Probability: %.2f%%" % prob)

    cv2.imwrite(img_path, orig)

