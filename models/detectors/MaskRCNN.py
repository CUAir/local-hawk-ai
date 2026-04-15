# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
# from typing import List
# import os
# import numpy as np
# import torch

# from constructs.roi import ROI
# from vision.detectors.abstract_detector import AbstractDetector

# SCORE_THRESH = 0.5
# CONFIG_FILE = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), "maskrcnn_config.yaml"
# )

# MODEL_WEIGHTS_FILE = os.path.join("model_weights", "maskrcnn_Mar52025.pth")

# import detectron2.utils.logger
# detectron2.utils.logger.setup_logger(name=__name__)


# class MaskRCNN(AbstractDetector):
#     def __init__(
#         self,
#         model_weights_file=MODEL_WEIGHTS_FILE,
#         score_thresh=SCORE_THRESH,
#         config_file=CONFIG_FILE,
#         use_gpu=False,
#     ):
#         self.cfg = get_cfg()
#         self.cfg.merge_from_file(config_file)
#         self.cfg.MODEL.WEIGHTS = model_weights_file
#         print(f"Model loaded from {model_weights_file}")
#         self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
#         if not use_gpu:
#             self.cfg.MODEL.DEVICE = "cpu"

#         print("Creating predictor...")
#         self.predictor = DefaultPredictor(self.cfg)
#         print("Predictor created successfully")

#     def detect(self, image) -> List[ROI]:
#         """
#         Detects targets in the image and returns a list of ROIs.

#         Args:
#             image: The image to detect targets in.

#         Returns:
#             A list of ROIs.
#         """
#         print(f"Input image shape: {np.array(image).shape}")
        
#         np_img = np.array(image)

#         output = None
#         print("Starting Detection...")
#         try:
#             output = self.predictor(np_img)
#         except Exception as e:
#             print(e)
        
#         # return [ROI(image.crop([1000, 1000, 2000, 2000]), (1000, 1000), (2000, 2000))]

#         rois = []
#         print(f"Processing {len(output['instances'].pred_boxes)} detections")
#         for box in output["instances"].pred_boxes:
#             x1, y1, x2, y2 = [int(coord) for coord in box]
#             cropped_img = image.crop([x1, y1, x2, y2])
#             # mask = mask.detach().cpu().numpy()[y1:y2, x1:x2]
#             rois.append(ROI(cropped_img, (x1, y1), (x2, y2)))

#         print(f"Returning {len(rois)} ROIs")
#         return rois
    
    
