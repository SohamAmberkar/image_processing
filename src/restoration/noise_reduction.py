import cv2

def denoising(image, kernel_size=(5, 5), sigma=0.0):
    """
    Apply noise reduction to the input image using a Gaussian blur.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the Gaussian kernel (default: (5, 5)).
        sigma (float): The standard deviation of the Gaussian kernel (default: 0.0).

    Returns:
        numpy.ndarray: The noise-reduced image.
    """
    # Apply the Gaussian blur
    denoised_image = cv2.GaussianBlur(image, kernel_size, sigma)

    return denoised_image
