import torch
import os
from torchvision import models, transforms
from constructs.classification import LabelType, Classification
from vision.classifiers.abstract_classifier import AbstractClassifier
from PIL import Image

class ImageNetNumberClassifier(AbstractClassifier):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.class_names = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']


        model_weights_path = os.path.join(
            "model_weights", "number_classifier_weights_resnet18.pt"
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 11)
        self.model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
    def classify_single_image_from_folder(self, folder_path):
            """
            Loads exactly one image from a folder and classifies it.

            Args:
                folder_path (str): Path to the folder containing one image.

            Returns:
                Tuple[Number, float]: Predicted label and confidence.
            """
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            assert len(image_files) == 1, f"Expected exactly one image, found {len(image_files)}."

            image_path = os.path.join(folder_path, image_files[0])
            img = Image.open(image_path).convert("RGB")

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                max_prob, predicted_idx = torch.max(outputs, 1)
                confidence = max_prob.item()
                predicted_label = self.class_names[predicted_idx.item()]

            print("✅ Folder Image Prediction:", predicted_label)
            return Number(int(predicted_label)), confidence

    

    def classify(self, roi):
        """
        Classifies the number of the target in the ROI.

        Args:
            roi: The ROI to classify.

        Returns:
            The tuple of the number of the target in the ROI and the confidence.
        """
        img = roi.image.convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        print("Classifying for Number...")
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            max_prob, predicted_idx = torch.max(outputs, 1)
            confidence = max_prob.item()
            predicted_label = self.class_names[predicted_idx.item()]

        print("PREDICTED NUMBER: ", predicted_label)
        return Number(int(predicted_label)), confidence
    
class FilterClassifier(AbstractClassifier):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        model_weights_path = os.path.join(
            "model_weights", "filter_classifier_weights.pt"
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

    def classify(self, roi):
        """
        Classifies the filter of the target in the ROI.
        """
        img = roi.image.convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        print("Classifying for Filter...")
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            max_prob, predicted = torch.max(outputs, 1)
            confidence = max_prob.item()

        print("PREDICTED FILTER: ", predicted.item())

        return predicted.item(), confidence
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "testimg")  # 👈 points to the testimg/ folder next to ImageNet.py
    classifier = ImageNetNumberClassifier()
    result = classifier.classify_single_image_from_folder(folder)
    print("🧠 Final result:", result)