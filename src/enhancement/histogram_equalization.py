import cv2

def histogram_equalization(image):
    """
    Apply histogram equalization to the input image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The histogram-equalized image.
    """
    # Convert the image to the YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization to the Y channel
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

    # Convert the image back to the BGR color space
    enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return enhanced_image
