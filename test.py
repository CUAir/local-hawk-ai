import os
from PIL import Image
from models.detectors.MaskRCNN import MaskRCNN
from models.classifiers.ImageNet import ImageNetNumberClassifier

# Path to the image
image_path = os.path.join("test_data/DJI_0726.JPG")

# Load the image
print(f"Loading image from {image_path}")
image = Image.open(image_path)
print(f"Image loaded, size: {image.size}")

# Initialize the detector
print("Initializing MaskRCNN detector...")
detector = MaskRCNN(use_gpu=False)

# Detect targets in the image
print("Detecting targets...")
rois = detector.detect(image)
print(f"Detected {len(rois)} targets")

# Initialize the classifier
print("Initializing ImageNet number classifier...")
classifier = ImageNetNumberClassifier()

# Classify each detected target
print("\nClassification results:")
for i, roi in enumerate(rois):
    number, confidence = classifier.classify(roi)
    print(f"Target {i+1}: Number {number.value} with confidence {confidence:.4f}")
    

print("\nDetection and classification completed!")
