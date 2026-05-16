# 🎧 AudioNode AI

AudioNode AI is an intelligent music analysis system that helps you understand songs beyond just sound. It combines machine learning and audio signal processing to identify a song’s **genre**, **musical key**, and **chord progression** — all from an uploaded audio file.

It is designed for musicians, learners, and developers who want deeper insights into how music is structured.

---

## ✨ What it does

When you upload a song, AudioNode AI breaks it down into meaningful musical elements:

- 🎼 Detects the **genre** of the song using a trained deep learning model  
- 🎹 Identifies the **musical key**  
- 🎧 Extracts and tracks **chords over time**  
- 📊 Provides **confidence scores and top predictions**  
- 🎤 Converts **frequencies into musical notes** for basic tuning insights  

---

## 🧠 How it works

AudioNode AI combines machine learning with signal processing:

- Audio is processed using **Librosa**
- Features like **MFCC, Chroma, and Spectral Contrast** are extracted  
- A trained neural network predicts the **genre**  
- A rule-based system analyzes harmony to detect **key and chords**  
- Results are returned through a simple **Flask API**

---

## ⚙️ Tech Stack

- Python  
- Flask  
- TensorFlow / Keras  
- Librosa  
- NumPy  
- SciPy  
- Scikit-learn  

---

## 📡 API Endpoints

### `/predict_audio` (POST)

Upload an audio file and get:

- Predicted genre  
- Confidence score  
- Top 3 genre predictions  
- Detected key  
- Chord progression timeline  
- Waveform data  

---

### `/predict_frequency` (POST)

Send a frequency value and receive:

- Closest musical note  
- Suggested chords  

---

## 🎯 Use Cases

- 🎸 Music learning apps  
- 🎧 Audio analysis tools  
- 🎼 Music education platforms  
- 🎬 Content creation and mood detection  
- 🎹 Audio intelligence systems  

---

## 🚀 Future Improvements

- Real-time microphone-based analysis  
- Live chord visualization interface  
- CNN-based spectrogram model for better accuracy  
- Mood and emotion detection from music  
- Interactive web UI  

---

## 📁 Project Structure

AudioNode-AI/

  │── app.py
  
  │── model.weights.h5
  
  │── scaler.pkl
  
  │── label_encoder.pkl
  
  │── templates/
      |── index.html
      
  │── static/
    |── style.css


## 💡 Closing Note

AudioNode AI is an ongoing exploration into how machines can understand music not just as sound, but as structure, emotion, and pattern.
