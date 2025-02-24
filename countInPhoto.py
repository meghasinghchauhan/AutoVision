import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

def count_cars_in_photo(image_path):
    """Detects and counts cars in an image."""
    im = cv2.imread(image_path)
    bbox, label, conf = cv.detect_common_objects(im)
    output_image = draw_bbox(im, bbox, label, conf)
    return output_image, label.count('car')
