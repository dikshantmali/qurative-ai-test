from jinja2 import Template
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.xception import preprocess_input
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)


# img_size = (299,299)
# preprocess_input = keras.applications.xception.preprocess_input
# last_conv_layer_name = "block14_sepconv2_act"

# # Loading our trained model for Atelectasis
# atelectasis_model = load_model('static/all models/Atelectasis.h5' , compile=True)

# picfolder = os.path.join('static','upload')
# app.config['UPLOAD_FOLDER'] = picfolder

# picfolder1 = os.path.join('static','uploadseparate')
# app.config['SEPARATE_UPLOAD_FOLDER'] = picfolder1

# path = "static/upload"
# path1 = "static/uploadseparate"
# start of gradcam
'''
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    '''

#endof gradcam



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/aboutus')
def About():
    return render_template('aboutus.html')


@app.route('/dev')
def dev():
    return render_template('devTeam.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/separatetest')
def separatetest():
    return render_template('separatetest.html')



# def model_predict_for_atelactasis(img_path, model):
#     img_array = preprocess_input(get_img_array(img_path, size=img_size))
#     preds = model.predict(img_array)
#     i = np.argmax(preds[0])
#     if i==0:
#         return "Atelectasis"
#     else:
#         return "Normal"


# @app.route('/uploadingimage' ,  methods=['GET','POST'])
# def uploadingimage():
#     if request.method == 'POST':
#         f = request.files['chest-x-ray']
#         file1_path = os.path.join(path,secure_filename(f.filename))
#         f.save(file1_path)
#         pic1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
#         # Make Prediction
#         atelectasis_result = model_predict_for_atelactasis(file1_path, atelectasis_model)
#         return render_template('upload.html' , atelectasis_result = atelectasis_result
         
#          )

# @app.route('/separateupload' ,  methods=['GET','POST'])
# def separateupload():
#     if request.method == 'POST':
#         f = request.files['chest-x-ray']
#         file1_path = os.path.join(path1,secure_filename(f.filename))
#         f.save(file1_path)
#         return render_template('separatetest.html')


if __name__ == '__main__':
    app.run(debug=True)
