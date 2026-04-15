# Onboarding Task: Swapping MaskRCNN Detection with Grounding DINO

## Problem Statement

Our current vision pipeline uses MaskRCNN for object detection, which detects targets in images and returns regions of interest (ROIs). These ROIs then go through:

1. **Filter Classification** - Determines if the ROI contains a valid target
2. **Number Classification** - If valid, classifies the number on the target

Currently, the detection is implemented using MaskRCNN (`models/detectors/MaskRCNN.py`), which requires:
- Pre-trained model weights
- Detectron2 framework
- Specific configuration files

**The Task**: Replace MaskRCNN detection with Grounding DINO, a zero-shot object detection model that can detect objects based on text prompts. This will allow us to:
- Use prompt-based detection ("tent. person.") instead of pre-trained classes
- Leverage Grounding DINO's powerful zero-shot capabilities
- Maintain the same ROI output format for seamless integration with existing classification pipeline

## Your Task

### Step 1: Research Grounding DINO

Before implementing, research and understand:

1. **What is Grounding DINO?**
   - How does it work?
   - What are zero-shot object detection capabilities?
   - How does text prompting work?

2. **Grounding DINO Installation**
   - What packages are required?
   - How do you install Grounding DINO?
   - What are the dependencies?

3. **Grounding DINO API**
   - How do you initialize the model?
   - How do you perform inference with text prompts?
   - What format are the bounding box outputs?
   - How do you convert outputs to ROI format?

4. **Comparison with MaskRCNN**
   - What are the differences in output format?
   - How do bounding boxes differ?
   - What are the performance considerations?

### Step 2: Analysis Questions

After your research, answer these questions:

1. **How does Grounding DINO output format differ from MaskRCNN?**
2. **What preprocessing/postprocessing is needed for the bounding boxes?**
3. **How will you handle the text prompt "tent. person."?**
4. **What various versions of Grounding DINO exist and which should we start with?**

### Step 3: Implementation

Create a new detector class `GroundingDINO` in `models/detectors/GroundingDINO.py` that:

1. **Implements the `AbstractDetector` interface**
   - Follows the same interface as `MaskRCNN` in `vision/detectors/abstract_detector.py`
   - Has a `detect(image)` method that returns `List[ROI]`

2. **Initializes Grounding DINO model**
   - Loads the Grounding DINO model and weights
   - Configures the model for inference
   - Handles device selection (CPU/GPU)

3. **Performs detection with prompt**
   - Takes an image as input
   - Uses the text prompt "tent. person." for detection
   - Returns bounding boxes with confidence scores

4. **Converts outputs to ROI format**
   - Extracts bounding box coordinates (x1, y1, x2, y2)
   - Crops image regions based on bounding boxes
   - Creates `ROI` objects compatible with existing classification pipeline

### Step 4: Modify Core Pipeline

Update `core.py` to use the new detector:

1. **Import GroundingDINO**
   - Add import for the new detector class

2. **Update VisionClient initialization**
   - Replace MaskRCNN with GroundingDINO in the `__init__` method
   - Ensure the detector follows the same interface

3. **Test the integration**
   - The `run_model` method should work without changes
   - Classification pipeline should receive ROIs in the same format

### Step 5: Testing

Create a test script to verify functionality:

1. **Test GroundingDINO detector**
   - Load a test image
   - Run detection with "tent. person." prompt
   - Verify bounding boxes are returned correctly
   - Check ROI format matches expected structure

2. **Test end-to-end pipeline**
   - Run full pipeline with GroundingDINO
   - Verify detection ? classification flow works
   - Compare outputs with MaskRCNN (if possible)

3. **Test edge cases**
   - Images with no detections
   - Images with multiple detections
   - Different image sizes and formats

### Step 6: Documentation

Document your solution:
- **Why** you chose Grounding DINO
- **How** the implementation works
- **How** to install and configure Grounding DINO
- **How** the prompt "tent. person." is used
- **Any** differences or considerations compared to MaskRCNN

## Success Criteria

- [ ] Researched Grounding DINO and its capabilities
- [ ] Answered the analysis questions
- [ ] Created `GroundingDINO` detector class implementing `AbstractDetector`
- [ ] Implemented detection with "tent. person." prompt
- [ ] Converted bounding boxes to ROI format correctly
- [ ] Updated `core.py` to use GroundingDINO
- [ ] Created test script verifying functionality
- [ ] Tested end-to-end pipeline integration
- [ ] Code is clean and well-documented

## Resources

- [Grounding DINO GitHub Repository](https://github.com/IDEA-Research/GroundingDINO)
- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [Zero-shot Object Detection](https://en.wikipedia.org/wiki/Zero-shot_learning)
- [AbstractDetector Interface](vision/detectors/abstract_detector.py)
- [MaskRCNN Reference Implementation](models/detectors/MaskRCNN.py)

## Getting Started

1. Clone the repository with `git clone https://github.com/CUAir/hawk-ai.git`
2. Create a new branch: `git checkout -b grounding-dino`
3. Start with the research phase
4. Check in with Justin or an Intsys sophomore
5. Implement your solution
6. Test using images in `test_data/` directory (add new aerial images of people and tends from Google)
7. Submit a pull request with your implementation

Good luck! This is a great opportunity to learn about Git, Hawk-AI's structure, zero-shot object detection, and integrating new models into existing pipelines.
