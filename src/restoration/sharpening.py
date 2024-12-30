import cv2

def sharpening(image, kernel_size=(3, 3), sigma=1.0):
    """
    Apply sharpening to the input image.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the Gaussian kernel (default: (3, 3)).
        sigma (float): The standard deviation of the Gaussian kernel (default: 1.0).

    Returns:
        numpy.ndarray: The sharpened image.
    """
    # Create the sharpening kernel
    kernel = cv2.getGaussianKernel(kernel_size[0], sigma) * cv2.getGaussianKernel(kernel_size[1], sigma).T
    kernel = kernel / kernel.max()

    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
