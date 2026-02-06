# Facial Emotion Recognition - CNN Deep Learning Model

Deep learning model for real-time facial emotion recognition using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset.

## Project Overview

This project builds a complete emotion recognition system that:
- Trains a CNN to classify 7 facial emotions from grayscale images
- Handles severely imbalanced classes using class weighting
- Applies conservative data augmentation for realistic face variations
- Achieves 55% test accuracy on the challenging FER-2013 dataset
- Provides detailed performance analysis per emotion class

## Model Performance

### Overall Metrics
- **Test Accuracy:** 55.48%
- **Validation Accuracy:** 60.96%
- **Dataset:** FER-2013 (35,887 images)

### Performance by Emotion

| Emotion | Precision | Recall | F1-Score | Test Samples |
|---------|-----------|--------|----------|--------------|
| Happy | 0.85 | 0.77 | 0.81 | 1,774 |
| Surprise | 0.62 | 0.82 | 0.71 | 831 |
| Neutral | 0.49 | 0.61 | 0.54 | 1,233 |
| Sad | 0.47 | 0.37 | 0.42 | 1,247 |
| Angry | 0.46 | 0.49 | 0.48 | 958 |
| Fear | 0.38 | 0.15 | 0.21 | 1,024 |
| Disgust | 0.16 | 0.77 | 0.27 | 111 |

**Key Insights:**
- Model excels at recognizing positive emotions (Happy, Surprise)
- Struggles with subtle emotions (Fear, Disgust) due to class imbalance
- Disgust is the most challenging class (only 111 test samples)

## Dataset

### FER-2013 Dataset
- **Total Images:** 35,887 grayscale facial images
- **Image Size:** 48×48 pixels
- **Classes:** 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Split:** 
  - Training: 24,399 images (68%)
  - Validation: 4,310 images (12%)
  - Test: 7,178 images (20%)

### Class Distribution Challenges
The dataset has severe class imbalance:
- **Most common:** Happy (~7,215 samples)
- **Least common:** Disgust (~436 samples)
- **Imbalance ratio:** 16.5:1

**Solution:** Implemented class weights (disgust weighted 7.8x higher than happy during training)

## Model Architecture

### CNN Design
```
Input (48×48×1 grayscale)
↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
↓
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
↓
Dense(256) → BatchNorm → Dropout(0.5)
↓
Dense(7, softmax)
```

**Architecture Highlights:**
- **Double convolutions per block:** Better feature extraction
- **Batch Normalization:** Stabilizes training, enables higher learning rates
- **Progressive filter increase:** 32 → 64 → 128 → 256 filters
- **Aggressive dropout (0.5):** Prevents overfitting on imbalanced data
- **Padding='same':** Preserves spatial dimensions for better feature retention

### Training Configuration
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 64 images
- **Epochs:** 100 (with early stopping)
- **Class Weighting:** Enabled to handle imbalance

### Data Augmentation Strategy

**Conservative augmentation for realistic faces:**
- **Rotation:** ±15° (natural head tilt)
- **Width/Height Shift:** 10% (off-center faces)
- **Zoom:** ±10% (different camera distances)
- **Horizontal Flip:** Yes (emotions look same when mirrored)
- **No vertical flip:** Upside-down faces aren't realistic
- **No shear:** Distorts facial proportions unnaturally

## Tech Stack

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **scikit-learn** - Metrics and validation
- **OpenCV** - Image processing (future deployment)

## Project Structure
```
facial-emotion-recognition/
├── facial_emotion_recognition.ipynb  # Complete pipeline notebook
├── models/
│   └── fer_emotion_model_final.h5   # Trained model weights
├── results/
│   ├── confusion_matrix.png         # Per-class performance
│   └── training_history.png         # Learning curves
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- GPU recommended (training takes ~2 hours on GPU, 8+ hours on CPU)
- 12GB+ RAM for full dataset loading

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/noelle-mburu/facial-emotion-recognition.git
cd facial-emotion-recognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download FER-2013 dataset**
- Download from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Extract to create `data/` folder with structure:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Usage

### Training the Model

**Option 1: Run complete notebook**
```bash
jupyter notebook facial_emotion_recognition.ipynb
```

**Option 2: Train from scratch (Python script)**
```python
# Load and preprocess data
python preprocess.py

# Train model
python train.py --epochs 100 --batch_size 64

# Evaluate
python evaluate.py --model models/fer_emotion_model_final.h5
```

### Using Pre-trained Model
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('models/fer_emotion_model_final.h5')

# Predict on new image
img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = img.reshape(1, 48, 48, 1) / 255.0

prediction = model.predict(img)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
result = emotions[np.argmax(prediction)]
print(f"Detected emotion: {result}")
```

## Results & Analysis

### Training Progress
- **Initial accuracy:** ~15% (random baseline: 14%)
- **Peak validation accuracy:** 60.96% at epoch 55
- **Final test accuracy:** 55.48%
- **Training time:** ~30 minutes on Google Colab GPU

### Confusion Matrix Analysis
The confusion matrix reveals:
- **Happy → Happy:** 77% correct (best performing class)
- **Surprise → Surprise:** 82% recall (highly recognizable)
- **Fear misclassified as:** Sad (28%), Neutral (22%)
- **Disgust low precision:** Frequently confused with Angry

### Model Limitations
1. **Fear detection:** Only 15% recall due to subtle features
2. **Disgust recognition:** Severe class imbalance (16.5:1 ratio)
3. **Dataset quality:** FER-2013 has ~9% label noise (known issue)
4. **Grayscale limitation:** Missing color cues (skin tone, context)

## Pipeline Phases

### Phase 1: Data Exploration
- Analyzed class distribution
- Identified 16.5:1 imbalance ratio
- Verified image quality (48×48 grayscale, face-cropped)

### Phase 2: Preprocessing
- Created 70/15/15 train/val/test split
- Calculated class weights (disgust=7.8x, happy=0.47x)
- Configured conservative data augmentation

### Phase 3: Model Development
- Built CNN with 4 convolutional blocks
- Applied batch normalization for training stability
- Implemented progressive dropout (0.25 → 0.5)

### Phase 4: Training
- Used Adam optimizer with learning rate decay
- Monitored validation loss for early stopping
- Saved best model checkpoint (epoch 55)

### Phase 5: Evaluation
- Generated confusion matrix
- Calculated per-class precision/recall/F1
- Analyzed misclassification patterns

## Challenges Solved

### Technical Challenges
- **Memory constraints:** Used data generators instead of loading full dataset
- **Training instability:** Added batch normalization and gradient clipping
- **Overfitting:** Applied dropout and early stopping
- **Class imbalance:** Implemented weighted loss function

### Dataset Challenges
- **Severe imbalance:** 16.5:1 ratio between most/least common classes
- **Small image size:** 48×48 limits detail for subtle emotions
- **Label noise:** ~9% incorrect labels (documented FER-2013 issue)
- **Grayscale only:** Missing color information for emotion cues

## Future Improvements

### Model Enhancements
- Try transfer learning (VGG, ResNet pre-trained on faces)
- Implement attention mechanisms for facial regions
- Ensemble multiple models for better accuracy
- Add focal loss for extreme class imbalance

### Dataset Improvements
- Use RAF-DB or AffectNet (higher quality, larger)
- Apply semi-supervised learning on unlabeled faces
- Implement active learning to fix mislabeled samples
- Augment disgust/fear classes with synthetic data

### Deployment
- Build real-time webcam emotion detection
- Deploy as Flask/FastAPI REST API
- Create Streamlit web interface
- Optimize for mobile (TensorFlow Lite)

## Research Context

### FER-2013 Benchmark Results
This project's 55% test accuracy is within expected range:
- **Human performance:** ~65% (due to label noise)
- **State-of-the-art (2024):** ~73% (ensemble + attention)
- **Baseline CNN:** ~50-60%

### Why FER-2013 is Challenging
- Small images (48×48 vs typical 224×224)
- Grayscale only (missing color context)
- Wild-collected (varied lighting, poses, occlusions)
- Label noise (~9% incorrect annotations)

## Contributing

This is a portfolio project, but feedback is welcome! Feel free to:
- Report issues
- Suggest improvements
- Share your own experiments

## Contact

**Noelle Mburu**
- GitHub: [@noelle-mburu](https://github.com/noelle-mburu)
- LinkedIn: www.linkedin.com/in/noelle-mburu
- Email: noellemburu@gmail.com

## License

This project is open source under the MIT License.

## Acknowledgments

- **FER-2013 Dataset** 
- **Google Colab**
- **Keras Team**

## References

1. Goodfell, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests."
2. FER-2013 Kaggle Competition: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge


**Built with ❤️ for learning deep learning fundamentals and computer vision techniques**