from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
import numpy as np
import cv2
import os
import torch
import base64
from io import BytesIO
from typing import Tuple, List
from enum import Enum
from constructs.image_types import Base64Image, CandidateImage, LabelTypes


# Text prompt given to DINO.  Periods separate distinct concepts.
DINO_PROMPT = (
    "person . mannequin . human . "
    "tent . shelter . canopy . tarp . blue object . square . box"
)

# DINO thresholds — intentionally very low to maximise recall.
# The VLM (Gemini) is the real filter; DINO is just a proposal generator.
DINO_BOX_THRESH = 0.12
DINO_TEXT_THRESH = 0.10

"""
Class to handle detecting with grounding dino. Consists of a function to return the bounding boxes from an image.
"""
class GDDetection:
    def __init__ (self):
        """
        Initialize the model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(
            os.path.join(script_dir, "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            os.path.join(script_dir, "../GroundingDINO/weights/groundingdino_swint_ogc.pth"),
            device=self.device,
        )
    
    def _load_image_b64(self, image_b64: str) -> Tuple[np.array, torch.Tensor]:
        """ Load the image from the base64 string for GroundingDINO analysis """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_data = base64.b64decode(image_b64)
        bytes_buffer = BytesIO(image_data)

        image_source = Image.open(bytes_buffer).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    
    def detect_b64(self, image_b64 : str, save_file : bool = False):
        """
        Return the bounding boxes and corresponding labels/phrases predicted by the Grounding dino boxes. 
        
        :param image_path: file path to the image
        """
        image_source, image = self._load_image_b64(image_b64) # Load image

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=DINO_PROMPT,
            box_threshold=DINO_BOX_THRESH,
            text_threshold=DINO_TEXT_THRESH,
            device=self.device,
        )

        # To save an image annotated with the bounding boxes
        if save_file:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite("annotated_image.jpg", annotated_frame)
        
        # Done in the analysis function in the multiprocessing file.
        # Change box to be based on the width and height of image instead of the absolute value.
        # h, w, _ = image_source.shape
        # boxes = boxes * torch.Tensor([w, h, w, h])
        
        return boxes, logits, phrases, image_source

    def _phrase_to_label(self, phrase: str) -> LabelTypes:
        """Map GroundingDINO phrases into repo label types."""
        normalized = (phrase or "").lower()

        if any(token in normalized for token in ("tent", "shelter", "canopy", "tarp")):
            return LabelTypes.TENT
        if any(token in normalized for token in ("mannequin", "person", "human")):
            return LabelTypes.MANNEQUIN
        return LabelTypes.UNKNOWN

    def _bbox_to_xyxy(
        self,
        bbox,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """Convert normalized center-format bbox into absolute corner-format."""
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2) * image_width))
        y1 = max(0, int((cy - bh / 2) * image_height))
        x2 = min(image_width, int((cx + bw / 2) * image_width))
        y2 = min(image_height, int((cy + bh / 2) * image_height))
        return x1, y1, x2, y2

    def detect_candidates(
        self,
        image: Base64Image,
        max_box_fraction: float = 0.5,
        save_file: bool = False,
    ) -> List[CandidateImage]:
        """Run GroundingDINO and return filtered `CandidateImage` results."""
        boxes, logits, phrases, image_source = self.detect_b64(
            image.base64_image,
            save_file=save_file,
        )

        image_height, image_width, _ = image_source.shape
        candidates: List[CandidateImage] = []

        for bbox, logit, phrase in zip(boxes, logits, phrases):
            x1, y1, x2, y2 = self._bbox_to_xyxy(bbox, image_width, image_height)

            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)
            if box_width == 0 or box_height == 0:
                continue

            if (box_width / image_width) > max_box_fraction or (box_height / image_height) > max_box_fraction:
                continue

            candidates.append(
                CandidateImage(
                    bbox=[x1, y1, x2, y2],
                    score=float(logit),
                    label=self._phrase_to_label(phrase),
                    source=image,
                )
            )

        return candidates

# Testing
# gdm = GDDetection()
# b64 = "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"
# gdm.detect_b64(image_b64 =b64, save_file=True)
