import cv2
from src.data_preprocessing import load_images_from_folder, save_image
from src.enhancement.histogram_equalization import histogram_equalization
from src.enhancement.contrast_adjustment import contrast_adjustment
from src.restoration.noise_reduction import denoising
from src.restoration.sharpening import sharpening
from src.segmentation import kmeans_segmentation
from src.utils import show_image
from src.captioning import describe_image 

def main():
    # Load images
    images = load_images_from_folder('data/raw/')
    
    for i, image in enumerate(images):
        # Apply image restoration
        #restored_image = denoising(image)
        #restored_image = sharpening(image)
        
        # Apply image enhancement
        enhanced_image = histogram_equalization(image)
        enhanced_image = contrast_adjustment(enhanced_image)
        
        # Apply K-means segmentation
        segmented_image = kmeans_segmentation(enhanced_image, k=3)
        
        # Show original, restored, enhanced, and segmented images
        show_image(image, title="Original Image")
        #show_image(restored_image, title="Restored Image")
        show_image(enhanced_image, title="Enhanced Image")
        show_image(segmented_image, title="Segmented Image")
        
        # Save the restored, enhanced, and segmented images
        #save_image(restored_image, f'data/processed/restored_{i}.png')
        save_image(enhanced_image, f'data/processed/enhanced_{i}.png')
        save_image(segmented_image, f'data/processed/segmented_{i}.png')

        description = describe_image(f'data/processed/enhanced_{i}.png')
        print(f"Description for segmented image {i}: {description}")

if __name__ == "__main__":
    main()
