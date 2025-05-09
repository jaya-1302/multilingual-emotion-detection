
# Multilingual Emotion Detection in Voice

This project focuses on detecting emotions from voice recordings across multiple languages using machine learning and deep learning techniques.

## 📌 Project Overview

The goal is to classify emotions such as *happy, sad, angry, neutral*, etc., from audio speech signals, leveraging multilingual datasets and applying feature extraction techniques and neural networks.

## 🔍 Features

- Preprocessing of audio data (e.g., MFCC extraction)
- Multilingual dataset support
- Deep learning models (e.g., CNN, LSTM)
- Emotion classification
- Model evaluation and visualization

## 🧰 Technologies Used

- Python
- Jupyter Notebook
- Librosa (audio processing)
- Scikit-learn
- TensorFlow / Keras or PyTorch (depending on the notebook)
- Pandas, NumPy, Matplotlib

## 📁 Project Structure

```
multilingual_emotion_detection_in_voice.ipynb
data/                    # Directory for audio datasets
models/                  # Saved model files (optional)
README.md
requirements.txt         # Python dependencies (optional)
```

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure datasets are available in the correct directory structure.
4. Open the notebook:
   ```bash
   jupyter notebook multilingual_emotion_detection_in_voice.ipynb
   ```
5. Run all cells in sequence to train and evaluate the model.

## 📊 Datasets

This project can be used with publicly available datasets such as:

- RAVDESS (English)
- Emo-DB (German)
- CREMA-D (American English)
- TESS (Canadian English)

Make sure to adjust preprocessing steps if your dataset differs.

## 🧠 Model Training

Models are trained using extracted MFCCs and other relevant features from audio. Performance is evaluated using accuracy, confusion matrix, and other classification metrics.

## 📈 Evaluation

- Confusion Matrix
- Accuracy/Precision/Recall
- Visualization of training history

## 📬 Contact

For questions or contributions, please open an issue or pull request.
