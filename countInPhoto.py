import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

def count_cars_in_photo(image_path):
    """Detects and counts cars in an image."""
    im = cv2.imread(image_path)
    bbox, label, conf = cv.detect_common_objects(
    im,
    model='yolov3-tiny',
    config='model/yolov3-tiny.cfg',
    weights='model/yolov3-tiny.weights',
    labels='model/coco.names'
)
    output_image = draw_bbox(im, bbox, label, conf)
    return output_image, label.count('car')
