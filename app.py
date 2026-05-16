from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import os
import scipy.ndimage
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

app = Flask(__name__)

# ===== LOAD MODEL (MATCH TRAINING EXACTLY) =====
model = Sequential([
    Flatten(input_shape=(58,)),
    Dropout(0.2),

    Dense(512, activation='relu'),
    Dropout(0.2),

    Dense(256, activation='relu'),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(10, activation='softmax')
])

# ✅ Load weights
model.load_weights("model.weights.h5")

# ===== LOAD SCALER (CORRECT FILE) =====
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===== LOAD LABEL ENCODER =====
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# ===== CHORD TEMPLATE =====
def build_chord_templates():
    templates = {}

    for i, root in enumerate(notes):

        v = np.zeros(12)
        v[i], v[(i+4)%12], v[(i+7)%12] = 1,1,1
        templates[root] = v

        v = np.zeros(12)
        v[i], v[(i+3)%12], v[(i+7)%12] = 1,1,1
        templates[root+"m"] = v

        v = np.zeros(12)
        v[i], v[(i+4)%12], v[(i+7)%12], v[(i+10)%12] = 1,1,1,1
        templates[root+"7"] = v

        v = np.zeros(12)
        v[i], v[(i+4)%12], v[(i+7)%12], v[(i+11)%12] = 1,1,1,1
        templates[root+"maj7"] = v

        v = np.zeros(12)
        v[i], v[(i+3)%12], v[(i+7)%12], v[(i+10)%12] = 1,1,1,1
        templates[root+"m7"] = v

        v = np.zeros(12)
        v[i], v[(i+3)%12], v[(i+6)%12] = 1,1,1
        templates[root+"dim"] = v

    return templates


chord_templates = build_chord_templates()


def detect_chord(chroma_vector):
    chroma_vector = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-6)

    best_chord = None
    best_score = -1

    for chord, template in chord_templates.items():
        template = template / (np.linalg.norm(template) + 1e-6)
        score = np.dot(chroma_vector, template)

        if score > best_score:
            best_score = score
            best_chord = chord

    return best_chord


def smooth_chords_light(timeline):
    smoothed = []

    for i in range(len(timeline)):
        if i > 0 and i < len(timeline)-1:
            prev = timeline[i-1]["chord"]
            next_ = timeline[i+1]["chord"]
            chord = prev if prev == next_ else timeline[i]["chord"]
        else:
            chord = timeline[i]["chord"]

        smoothed.append({
            "time": timeline[i]["time"],
            "chord": chord
        })

    return smoothed


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    try:
        file = request.files["audio"]
        path = "uploads/" + file.filename
        file.save(path)

        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # ===== FEATURE EXTRACTION =====
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma_feat = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spec = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        features = np.hstack([mfcc, chroma_feat, spec])

        # Ensure 58 features
        if len(features) > 58:
            features = features[:58]
        elif len(features) < 58:
            features = np.pad(features, (0, 58 - len(features)))

        features = features.reshape(1, -1)

        # ✅ APPLY SCALER
        features = scaler.transform(features)

        # ===== PREDICTION =====
        prediction = model.predict(features, verbose=0)[0]

        genre_index = np.argmax(prediction)

        # ✅ CORRECT LABEL MAPPING
        genre = label_encoder.inverse_transform([genre_index])[0]

        confidence = float(prediction[genre_index]) * 100

        # ===== TOP 3 =====
        top3_indices = np.argsort(prediction)[-3:][::-1]

        top3 = [
            {
                "genre": label_encoder.inverse_transform([i])[0],
                "confidence": round(float(prediction[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        # ===== CHORD DETECTION =====
        y_harmonic, _ = librosa.effects.hpss(y)
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chromagram = scipy.ndimage.median_filter(chromagram, size=(1,3))

        key_index = np.argmax(np.sum(chromagram, axis=1))
        song_key = notes[key_index]

        avg_chroma = np.mean(chromagram, axis=1)
        main_chord = detect_chord(avg_chroma)

        # ===== TIMELINE =====
        timeline = []

        hop_length = 512
        frames_per_segment = 100
        frames = chromagram.shape[1]

        for i in range(0, frames, frames_per_segment):
            segment = chromagram[:, i:i+frames_per_segment]

            if segment.shape[1] == 0:
                continue

            chroma_vector = np.mean(segment, axis=1)
            chord = detect_chord(chroma_vector)

            time = round(i * hop_length / sr, 2)

            timeline.append({
                "time": time,
                "chord": chord
            })

        timeline = smooth_chords_light(timeline)

        # ===== WAVEFORM =====
        step = max(1, len(y)//2000)
        waveform = y[::step].tolist()

        return jsonify({
            "genre": genre,
            "confidence": f"{confidence:.2f}%",
            "top3": top3,
            "duration": duration,
            "key": song_key,
            "chords": [main_chord],
            "timeline": timeline,
            "waveform": waveform
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict_frequency", methods=["POST"])
def predict_frequency():
    data = request.get_json()
    freq = float(data["frequency"])

    note_map = {
        261:"C", 293:"D", 329:"E",
        349:"F", 392:"G", 440:"A", 493:"B"
    }

    closest = min(note_map.keys(), key=lambda x: abs(x-freq))
    note = note_map[closest]

    chords = [note, note+"m", note+"7", note+"maj7", note+"m7"]

    return jsonify({
        "note": note,
        "suggested_chords": chords
    })


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)