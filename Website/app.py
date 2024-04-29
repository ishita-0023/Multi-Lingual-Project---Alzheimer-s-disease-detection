from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

model = load_model('models\model.h5')

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (128, 128))  
    normalized_img = resized_img / 255.0  
    yhat = model.predict(np.expand_dims(normalized_img, 0))
    classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'] 
    predicted_class = classes[np.argmax(yhat)] 
    return predicted_class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        

        if file.filename == '':
            return 'No selected file'
        
        
        if file and allowed_file(file.filename):
            
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            
            
            file_path = os.path.join(uploads_dir, secure_filename(file.filename))
            file.save(file_path)
            
         
            result = model_predict(file_path, model)
            return result

    return 'Error occurred during prediction'


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5926)

