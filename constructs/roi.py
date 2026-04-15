from PIL import Image
from typing import Tuple

class ROI(object):
    """
    Class to store the ROI of a target.
    """
    def __init__(
        self, roi: Image.Image, top_left: Tuple[int, int], bottom_right: Tuple[int, int]
    ):
        self.image = roi
        self.top_left = top_left
        self.bottom_right = bottom_right

        # 4-tuple: Box = (Left, Top, Right, Bottom) a.k.a (x1, y1, x2, y2)
        self.box = self.top_left + self.bottom_right
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        self.center = (
            self.box[0] + (self.width // 2),
            self.box[1] + (self.height // 2),
        )
