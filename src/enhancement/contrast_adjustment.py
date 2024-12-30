import cv2

def contrast_adjustment(image, alpha=1.2, beta=0):
    """
    Adjust the contrast of the input image.

    Args:
        image (numpy.ndarray): The input image.
        alpha (float): The contrast adjustment factor (default: 1.2).
        beta (int): The brightness adjustment factor (default: 0).

    Returns:
        numpy.ndarray: The contrast-adjusted image.
    """
    # Apply the contrast adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image
