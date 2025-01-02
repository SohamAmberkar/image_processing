from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
from datetime import datetime
import logging
from src.data_preprocessing import load_images_from_folder, save_image
from src.enhancement.histogram_equalization import histogram_equalization
from src.enhancement.contrast_adjustment import contrast_adjustment
from src.restoration.noise_reduction import denoising
from src.restoration.sharpening import sharpening

app = Flask(__name__)

# Folder to save uploaded and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@app.route('/')
def home():
    logging.info('Serving the home page')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        logging.error('No selected file')
        return "No selected file", 400
    if file:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.info(f'Uploaded file: {file.filename}')

        # Redirect to processing page
        return redirect(url_for('process_image', filename=file.filename))
    logging.error('Unexpected error during file upload')
    return redirect(url_for('home'))

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_images = {}

    if request.method == 'POST':
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            logging.error('Error: Unable to load the image.')
            return "Error: Unable to load the image.", 400

        # Apply selected operations
        if 'histogram_equalization' in request.form:
            img = histogram_equalization(img)
            processed_images['Histogram Equalization'] = img
            logging.debug('Histogram equalization applied.')

        if 'contrast_adjustment' in request.form:
            img = contrast_adjustment(img)
            processed_images['Contrast Adjustment'] = img
            logging.debug('Contrast adjustment applied.')

        if 'noise_reduction' in request.form:
            img = denoising(img)
            processed_images['Noise Reduction'] = img
            logging.debug('Noise reduction applied.')

        if 'sharpening' in request.form:
            img = sharpening(img)
            processed_images['Sharpening'] = img
            logging.debug('Sharpening applied.')

        # Save processed images
        processed_images_urls = {}
        for key, value in processed_images.items():
            # Generate a safe filename without spaces
            processed_filename = f"{key.replace(' ', '_')}_{filename}"
            processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            save_image(processed_file_path, value)
            
            # Generate the URL using only the filename
            processed_images_urls[key] = url_for('serve_processed_image', filename=processed_filename)

        
        

        # Render the result template
        return render_template('result.html', filename=filename, original_image=url_for('serve_uploaded_image', filename=filename), processed_images=processed_images_urls)

    logging.info(f'Serving the process page for: {filename}')
    return render_template('process.html', filename=filename)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    # Clear the uploads folder
    for uploaded_file in os.listdir(app.config['UPLOAD_FOLDER']):
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        os.remove(uploaded_file_path)
        logging.info(f'Deleted uploaded file: {uploaded_file}')
    
    # Clear the processed folder
    for processed_file in os.listdir(app.config['PROCESSED_FOLDER']):
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file)
        os.remove(processed_file_path)
        logging.info(f'Deleted processed file: {processed_file}')
    
    # Redirect to the home page
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
