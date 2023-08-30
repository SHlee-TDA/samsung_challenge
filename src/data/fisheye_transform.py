"""
Fisheye transform implementation
"""

def fisheye_transform(image, center=None, distortion):
    """
    Fisheye transform maps a given pixel location (x,y) to (T(x), T(y)), where
    T(x) = x + (x - c_x) * d * sqrt{(x - c_x)^2 + (y-c_y)^2}
    T(y) = y + (y - c_y) * d * sqrt{(x - c_x)^2 + (y-c_y)^2}
    
        Args:
            - image (ndarray) : input image
            - center (tuple) : center of transformation. If it is None, set it as the center of the input image.
            - distortion: distortion factor d
    """
    