from flask import Flask, render_template, request, flash, redirect
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (150, 150)
weather_classes = ['lightning', 'rain', 'rainbow', 'sandstorm', 'snow']

# Load model once at startup
model = load_model('./model/weather_detection_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        return img_array, img

def create_prediction_plot(predictions, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(weather_classes, predictions)
    plt.title('Weather Classification Probabilities')
    plt.xlabel('Weather Types')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f"{os.path.splitext(filename)[0]}_plot.png"
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return 'uploads/' + plot_filename

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    confidence = None
    image_path = None
    plot_path = None
    class_probabilities = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{int(time.time())}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                
                # Process image and get predictions
                img_array, img = preprocess_image(filepath)
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)[0]
                
                # Get prediction results
                prediction = weather_classes[np.argmax(predictions)]
                confidence = float(np.max(predictions))
                
                # Create plot and save resized image
                plot_path = create_prediction_plot(predictions, unique_filename)
                class_probabilities = {weather_classes[i]: float(predictions[i]) 
                                     for i in range(len(weather_classes))}
                
                resized_filename = f"{os.path.splitext(unique_filename)[0]}_resized.jpg"
                resized_path = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
                img.save(resized_path)
                image_path = 'uploads/' + resized_filename
                
                # Clean up original file
                os.remove(filepath)
                    
            except Exception as e:
                flash(f'Prediction error: {str(e)}', 'error')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Unsupported file format. Please upload PNG, JPG, or JPEG.', 'error')
            return redirect(request.url)
    
    return render_template('index.html',
                         prediction=prediction,
                         confidence=confidence,
                         image_path=image_path,
                         plot_path=plot_path,
                         class_probabilities=class_probabilities)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)