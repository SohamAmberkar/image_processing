import cv2
import os
from src.data_preprocessing import load_images_from_folder, save_image
from src.enhancement.histogram_equalization import histogram_equalization
from src.enhancement.contrast_adjustment import contrast_adjustment
from src.segmentation import kmeans_segmentation
from src.utils import show_image
from src.captioning import describe_image

def main():
    # Load images from the raw folder
    images = load_images_from_folder('data/raw/')
    
    for i, image in enumerate(images):
        # Apply image enhancement (Histogram Equalization and Contrast Adjustment)
        enhanced_image = histogram_equalization(image)
        enhanced_image = contrast_adjustment(enhanced_image)
        
        # Apply K-means segmentation
        segmented_image = kmeans_segmentation(enhanced_image, k=3)
        
        # Show original, enhanced, and segmented images
        show_image(image, title="Original Image")
        show_image(enhanced_image, title="Enhanced Image")
        show_image(segmented_image, title="Segmented Image")
        
        # Save the enhanced and segmented images
        save_image(f'data/processed/enhanced_{i}.png', enhanced_image)
        save_image(f'data/processed/segmented_{i}.png', segmented_image)

        # Generate a description of the segmented image
        description = describe_image(f'data/processed/segmented_{i}.png')
        print(f"Description for segmented image {i}: {description}")

if __name__ == "__main__":
    main()
