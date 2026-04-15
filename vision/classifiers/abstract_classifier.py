from constructs.roi import ROI
from constructs.classification import Classification

class AbstractClassifier(object):
  def __init__(self):
    pass
  @classmethod
  def classify(roi : ROI) -> Classification:
    pass