import os
import numpy as np
from flask import Flask, render_template, request, flash
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename, redirect
from brain.inference import predict
import tensorflow as tf

UPLOAD_FOLDER = 'D:/PycharmProjects/DogBreedPredictor/static/'
ALLOWED_EXTENSIONS = {'png', 'jpg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('../DogBreedPredictor/model_fine_final.h5')
# model._make_predict_function()
graph = tf.get_default_graph()
# load class names
with open('../DogBreedPredictor/brain/classes.txt', 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pred_name = filename[:-4] + '_pred' + filename[-4:]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], pred_name))
            (class_name, prob) = predict(os.path.join(app.config['UPLOAD_FOLDER'], pred_name))
            # img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('prediction.html', name=class_name[10:], proba="{:.2f}%".format(prob), image=pred_name)
    return render_template('predict_breed.html')


if __name__ == '__main__':
    app.run(debug=True)


def predict(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # predict
    with graph.as_default():
        pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda y: y[1])
    (class_name, prob) = result[0]
    return class_name, prob
