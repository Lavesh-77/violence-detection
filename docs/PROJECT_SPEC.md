# Project Specification: Local Violence-Detection App

## Problem and goal

Create a simple local web app that analyzes a video for possible violence and reports both an overall result and short time segments that may contain violence.

## Target user and core flow

The user selects a local video in the web page, starts analysis, and reviews whether violence was detected along with flagged timestamps and confidence scores.

## MVP requirements

- Provide a simple web page that runs locally.
- Let the user select and analyze a local video, one video at a time.
- Use Python and OpenCV for video handling and analysis integration.
- Use an already-trained violence-detection model; do not train from scratch for the MVP.
- Report an overall violence result.
- Report short flagged time segments with a confidence score for each.
- Use a publicly hosted, RWF-2000-fine-tuned 3D ResNet-18 checkpoint. The selected checkpoint is `Abdullahtahir11/rwf2000-violence-detection-3dresnet` on Hugging Face (MIT-licensed model card/checkpoint; model card results are self-reported).
- Do not download RWF-2000 raw videos automatically: the dataset project page says they are not publicly distributed and requires approval for access.

## Proposed technical approach

- Python backend using OpenCV to read video frames and analyze short windows.
- A lightweight local web interface, with Streamlit as the proposed option.
- Download and cache the selected checkpoint on first use; load it through torchvision's 3D ResNet-18 implementation and run locally with CUDA when available, otherwise CPU.
- Analyze 16 sampled frames per overlapping 2.5-second window, advancing by 1 second. The threshold defaults to 50% and is user-adjustable; merge adjacent flagged windows into time segments.

## User experience

1. User selects a local video.
2. User starts analysis and can see that processing is underway.
3. The page displays the overall result and a list of flagged short time segments with confidence scores.

## Out of scope for the MVP

- Training a model from scratch.
- Creating an annotated copy of the input video.
- Processing multiple videos as a batch.

## Privacy and limitations

- Videos are selected from the user's local machine and are intended to be processed locally.
- Confidence scores are model estimates, not guarantees or definitive judgments.
- Timestamp precision depends on the chosen model and analysis-window size; flags may not identify exact incident boundaries.

## Implementation milestones

1. Verify RWF-2000 access and identify a usable pretrained checkpoint, including its source, compatibility, and download requirements.
2. Implement video reading and short-window scoring with OpenCV and the selected model.
3. Build the local web page for video selection, analysis status, overall result, and timestamp/confidence results.
4. Test supported video formats, timestamp behavior, and score output with available sample videos.

## Acceptance criteria

- The user can select and analyze a local video through the web page.
- Analysis returns an overall violence result.
- When the detector flags violence, the result includes short timestamp segments and confidence scores.
- The app uses a pretrained model and does not require training from scratch for normal MVP use.
- Model/data availability and limitations are documented; the app does not present confidence as certainty.

## Open questions and assumptions

- Assumption: “rwf2000ya” refers to the RWF-2000 dataset.
- The model card reports 90% accuracy and 0.9649 ROC-AUC on RWF-2000; these results are self-reported and are not a guarantee of performance on other footage.
- The 2.5-second window length, 1-second stride, and default 50% threshold are implementation choices for short flags and may need tuning based on testing.
- The specific supported video formats and hardware/runtime requirements remain to be validated.
