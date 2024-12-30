from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

def describe_image(image_path_or_url):
    """
    Takes an image file path or URL as input and generates a description for the image.
    
    Args:
        image_path_or_url (str): Path to the image file or a URL.
    
    Returns:
        str: Generated description of the image.
    """
    try:
        # Load the image
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        
        # Load the model and processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Preprocess the image
        inputs = processor(image, return_tensors="pt")
        
        # Generate a description
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)
        
        return description
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example Usage
# Local Image: describe_image("path/to/image.jpg")
# URL Image: describe_image("https://example.com/image.jpg")
