# Human Action Recognition using LRCN Model on UCF50 Dataset

## Overview

This project implements a human action recognition system using a Long-term Recurrent Convolutional Network (LRCN) model trained on a subset of the UCF50 dataset. The system can recognize actions in videos, either from pre-recorded files or directly from YouTube URLs.

## Features

- **Action Recognition**: Classifies human actions in videos into one of 7 categories:
  - HighJump
  - JugglingBalls
  - MilitaryParade
  - RockClimbingIndoor
  - SkateBoarding
  - Skijet
  - Swing

- **Model Architecture**: Combines CNN and LSTM layers to capture both spatial and temporal features
- **Video Processing**: Handles frame extraction, normalization, and sequence preparation
- **YouTube Integration**: Can download and process videos directly from YouTube

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- yt-dlp (for YouTube video downloads)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Run `2_ucf50_human_action_recognisation.py` to:
   - Download and preprocess the dataset
   - Train the LRCN model
   - Save the trained model and label encoders

### Testing with Videos

1. Use `video_testing.py` to:
   - Download YouTube videos
   - Perform action recognition on videos
   - Display predictions frame-by-frame

Example usage:
```python
# For single action prediction
predict_single_action(video_path, SEQUENCE_LENGTH)

# For continuous prediction on video
predict_on_video(input_video_path, output_video_path, model, SEQUENCE_LENGTH)
```

## Model Architecture

The LRCN model consists of:
1. TimeDistributed CNN layers for spatial feature extraction
2. LSTM layer for temporal sequence modeling
3. Dense layer with softmax activation for classification

## Data Preprocessing

- Frame resizing to 64x64 (training) or 112x112 (testing)
- Normalization (mean subtraction and standard deviation division)
- Sequence generation (20-25 frames per sequence)

## Results

The model achieves:
- Training accuracy: [Value from training history]
- Validation accuracy: [Value from training history]
- Test accuracy: [Value from evaluation]

## Files

- `2_ucf50_human_action_recognisation.py`: Training script
- `video_testing.py`: Video prediction script
- `class_mappings.pkl`: Class label mappings
- `label_encoder.pkl`: Trained label encoder
- `LRCN_MODEL_UCF50_2.keras`: Trained model file

## Limitations

- Currently supports only 7 action classes
- Requires videos with clear action sequences
- Performance may vary with different video qualities

## Future Improvements

- Expand to more action classes
- Implement real-time prediction
- Add confidence thresholding
- Improve model architecture for better accuracy

## Acknowledgments

- UCF50 dataset
- Kaggle for hosting the dataset
- TensorFlow/Keras for deep learning framework
