from flask import Flask, render_template,request
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from resizeimage import resizeimage
import os

app = Flask(__name__)

SEGMENT_FOLDER = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = SEGMENT_FOLDER

json_file = open("model_nc.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("model_weights.h5")

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')
@app.route("/predict", methods = ['GET','POST'])
def predict():

    if request.method == "POST":
        img = Image.open(request.files['file'].stream)

        resized_img = resizeimage.resize_cover(img, [256, 256])
        img_arr = np.asarray(resized_img, dtype=np.float32)
        img_arr = img_arr.reshape(1, 256, 256, 3)
        pred = model.predict(img_arr)
        pred = pred.reshape(256, 256, 3)
        img = Image.fromarray(pred, 'RGB')

        img.save('static/images/image.png')

        image = [i for i in os.listdir('static/images') if i.endswith('.png')][0]

        BASE_DIR = os.getcwd()
        dir = os.path.join(BASE_DIR, "static/images")
        for root, dirs, files in os.walk(dir):
            for file in files:
                path = os.path.join(dir, file)
                os.remove(path)

        return render_template("predict.html",user_image = image)


if __name__ == "__main__":
    app.run(debug=True)