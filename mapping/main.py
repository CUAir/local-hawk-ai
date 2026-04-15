import cv2
import numpy as np
import csv
from typing import List, Dict, Tuple
cv2.ocl.setUseOpenCL(False)

REFERENCE_ALTITUDE = 100
REFERENCE_LATITUDE = 42
REFERENCE_LONGITUDE = -76

def read_csv(csv_path: str) -> Tuple[Dict[str, Tuple[float, float, float, float]], float, float]:
    """
    Reads the center locations and directions from the CSV file.
    Also calculates mean latitude and longitude as reference points.
    
    Args:
    csv_path (str): Path to the CSV file containing center locations and directions.
    
    Returns:
    Tuple[Dict[str, Tuple[float, float, float, float]], float, float]: Dictionary mapping image filenames to their coordinates,
    altitude, and direction, plus mean lat/lon reference points.
    """
    metadata = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image = row['Image']
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            altitude = float(row['Altitude'])
            direction = float(row['Degrees_Clockwise_from_North'])
            metadata[image] = (lat, lon, altitude, direction)
    
    return metadata

def rotate_image(image: np.ndarray, rotation_angle: float) -> np.ndarray:
    """
    Rotates the image based on the given angle to align north to the top.
    
    Args:
    image (np.ndarray): The input image.
    rotation_angle (int): The rotation angle in degrees from north clockwise.
    
    Returns:
    np.ndarray: The rotated image.
    """
    # Convert clockwise angle to counter-clockwise for rotation
    rotation_angle = -rotation_angle
    
    if rotation_angle == 0:
        return image
    elif rotation_angle == 180 or rotation_angle == -180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Calculate the dimensions of the new image
        rows, cols = image.shape[:2]
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_cols = int((rows * sin) + (cols * cos))
        new_rows = int((rows * cos) + (cols * sin))
        
        # Adjust the rotation matrix
        M[0, 2] += (new_cols / 2) - cols/2
        M[1, 2] += (new_rows / 2) - rows/2
        
        # Perform the rotation with the new dimensions
        return cv2.warpAffine(image, M, (new_cols, new_rows))

def resize_image(image: np.ndarray, altitude: float) -> np.ndarray:
    """
    Zooms the image based on the given zoom factor.
    
    Args:
    image (np.ndarray): The input image.
    altitude (float): The altitude of the drone in meters.
    
    Returns:
    np.ndarray: The zoomed image.
    """
    if altitude == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)  # Return a 1x1 black image
    resize_factor = altitude / REFERENCE_ALTITUDE 
    return cv2.resize(image, None, fx=resize_factor, fy=resize_factor)

def coordinates_to_pixels(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[int, int]:
    """
    Converts the latitude/longitude coordinates to pixel coordinates with improved accuracy.
    """
    # Use more precise conversion factors
    PIXELS_PER_DEGREE_LAT = 3200000  # pixels per degree of latitude
    PIXELS_PER_DEGREE_LON = 2400000  # pixels per degree of longitude
    
    pixel_lat = (lat - ref_lat) * PIXELS_PER_DEGREE_LAT
    pixel_lon = (lon - ref_lon) * PIXELS_PER_DEGREE_LON
    pixel_x = int(pixel_lon)
    pixel_y = int(-pixel_lat)
    return (pixel_x, pixel_y)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the exposure and white balance of an image.
    
    Args:
    image (np.ndarray): The input image.
    
    Returns:
    np.ndarray: The normalized image.
    """
    # Convert to LAB color space for better color/lighting separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(4,4))
    l = clahe.apply(l)

    l = cv2.multiply(l, 0.75)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized

def stitch_images(processed_images: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Stitches together all the images in a given folder using their center locations and rotates them to align north.
    """
    if not processed_images:
        raise ValueError("No images found to stitch.")
    
    # Calculate pixel coordinates for each image
    for img_data in processed_images:
        pixel_x, pixel_y = coordinates_to_pixels(img_data['lat'], img_data['lon'], REFERENCE_LATITUDE, REFERENCE_LONGITUDE)
        h, w = img_data['dimensions']
        img_data['position'] = {
            'left': pixel_x - (w // 2),
            'right': pixel_x + (w - w // 2),
            'top': pixel_y - (h // 2),
            'bottom': pixel_y + (h - h // 2),
            'center_x': pixel_x,
            'center_y': pixel_y
        }
    
    # Calculate canvas dimensions
    min_x = min(img['position']['left'] for img in processed_images)
    max_x = max(img['position']['right'] for img in processed_images)
    min_y = min(img['position']['top'] for img in processed_images)
    max_y = max(img['position']['bottom'] for img in processed_images)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    # Create and fill the canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for img_data in processed_images:
        img = img_data['image']
        pos = img_data['position']

        # Calculate placement coordinates
        x1 = pos['left'] - min_x
        y1 = pos['top'] - min_y
        x2 = x1 + img.shape[1]
        y2 = y1 + img.shape[0]
        
        # Create mask for non-black pixels
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10
        
        # Calculate source image coordinates
        img_x1 = max(0, -x1)
        img_y1 = max(0, -y1)
        img_x2 = img.shape[1] - max(0, x2 - canvas_width)
        img_y2 = img.shape[0] - max(0, y2 - canvas_height)
        
        # Copy only non-black pixels with transparency
        region = mask[img_y1:img_y2, img_x1:img_x2]
        alpha = 0.9 # Opacity
        for c in range(3):
            canvas[y1:y2, x1:x2, c][region] = (
                alpha * img[img_y1:img_y2, img_x1:img_x2, c][region] + 
                (1 - alpha) * canvas[y1:y2, x1:x2, c][region]
            ).astype(np.uint8)
    
    return canvas