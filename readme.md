# Human Activity Recognition Using PAMAP2 Dataset

## Overview

This project implements a deep learning pipeline for human activity recognition (HAR) using the PAMAP2 wearable sensor dataset. The goal is to accurately classify 18 different physical activities (such as walking, running, sitting, and more) based on time-series data collected from multiple body-worn sensors.

## Dataset

- **Source:** PAMAP2 Physical Activity Monitoring Dataset
- **Sensors:** 3 IMUs (wrist, chest, ankle) and a heart rate monitor
- **Sampling Rate:** 100 Hz (IMUs), ~9 Hz (heart rate)
- **Activities:** 18 labeled activities, including daily, household, and sports movements
- **Subjects:** 9 individuals, each performing a protocol and optional activities

## Project Pipeline

### 1. Data Preprocessing
- Loaded and merged raw sensor data from all subjects and sessions
- Mapped original activity IDs to 18 contiguous class labels (0–17)
- Selected relevant features (acceleration, gyroscope, heart rate), excluding faulty sensor channels
- Filled missing values using forward and backward fill
- Segmented data into overlapping windows (128 timesteps, 50% overlap)
- Normalized features using StandardScaler
- Removed any windows containing NaNs
- Split data into 80% training and 20% test sets, stratified by activity

### 2. Model Development
- Built an LSTM-based neural network for sequence classification
- Used two LSTM layers with dropout for regularization
- Output layer with 18 units (one per activity), softmax activation

### 3. Training and Evaluation
- Trained the model on the training set with early stopping and checkpointing
- Achieved ~97% accuracy on the training set and ~96.75% on the test set

## Results

| Metric         | Value      |
|----------------|-----------|
| Training Acc.  | ~97%      |
| Test Acc.      | ~96.75%   |
| Classes        | 18        |

The model demonstrates strong performance in recognizing a wide range of human activities from wearable sensor data.

## How to Run

1. **Clone the repository**
2. **Install dependencies** (Python 3.10+, TensorFlow, scikit-learn, pandas, numpy)
3. **Place the PAMAP2 dataset** in the `data/PAMAP2_Dataset/` directory
4. **Run preprocessing:**  
   ```
   python src/preprocess.py
   ```
5. **Train the model:**  
   ```
   python src/train.py
   ```

## Project Structure

```
├── data/
│   └── PAMAP2_Dataset/
├── models/
│   └── best_model.h5
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── models.py
├── README.md
```

## Future Work

- Explore more advanced models (CNN-LSTM, Transformers)
- Deploy the model for real-time activity recognition on mobile devices
- Analyze per-activity and per-subject performance in detail
- Integrate additional sensor modalities or external datasets

***

