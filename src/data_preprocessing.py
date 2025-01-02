import cv2
import os

def load_images_from_folder(folder):
    images = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']  # Add more extensions as needed
    for filename in os.listdir(folder):
        # Check if the file has a valid image extension
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Unable to load image {filename}")
        else:
            print(f"Skipping non-image file: {filename}")
    return images

def save_image(path, image):
    # Validate path
    if not isinstance(path, str):
        raise ValueError(f"The 'path' must be a string, got {type(path)} instead.")
    
    # Validate image
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image object. Ensure it is a valid OpenCV image.")
    
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            raise IOError(f"Failed to create directory {directory}: {str(e)}")
    
    # Save image
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to save image to {path}")
    print(f"Image saved successfully to {path}")
