from constructs.roi import ROI
from PIL import Image
import typing

class AbstractDetector(object):
  def __init__(self):
    pass
  @classmethod
  def detect(image : Image.Image) -> typing.List[ROI]:
    pass