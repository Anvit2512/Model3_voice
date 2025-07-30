import os
import contextlib
import wave
import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import webrtcvad
from tensorflow.keras.models import load_model
import joblib
import warnings
import tempfile

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# --- Configuration ---
TARGET_SR = 16000
MODEL_PATH = "vocal_model.h5"
SCALER_PATH = "vocal_scaler.joblib"
FEATURES_PATH = "feature_names.joblib"

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Load Models and Scaler at Startup ---
# This is efficient as they are loaded only once when the app starts
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✅ Model, scaler, and feature list loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model files. The application will not work.")
    print(f"   Details: {e}")
    # In a real-world scenario, you might want the app to fail to start here.
    model, scaler, feature_names = None, None, None

# --- Feature Extraction Functions (Copied from your script) ---
# (I've omitted the functions for brevity, but you should copy ALL of them here)
# - preprocess_audio
# - extract_features
# ... (all your existing helper functions) ...
def preprocess_audio(input_path, target_sr=TARGET_SR):
    try:
        data, sr = librosa.load(input_path, sr=None, mono=False)
        if data.ndim > 1: data = data.mean(axis=0)
        if sr != target_sr: data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_processed_for_prediction.wav"
        sf.write(output_path, data, target_sr, subtype='PCM_16')
        return output_path
    except Exception as e:
        print(f"Error preprocessing {input_path}: {e}")
        return None

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]

        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        def read_wave(path):
            with contextlib.closing(wave.open(path, 'rb')) as wf:
                pcm_data, sample_rate = wf.readframes(wf.getnframes()), wf.getframerate()
                return pcm_data, sample_rate
        
        def frame_generator(frame_duration_ms, audio, sample_rate):
            n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
            offset = 0
            while offset + n < len(audio):
                yield audio[offset:offset + n]
                offset += n
        
        vad = webrtcvad.Vad(1)
        audio, sample_rate = read_wave(file_path)
        frames = list(frame_generator(30, audio, sample_rate))
        voiced_seconds = 0
        num_segments = 0
        if frames:
            for frame in frames:
                if vad.is_speech(frame, sample_rate):
                    voiced_seconds += 0.03 # 30ms frame
                    num_segments +=1

        silence_ratio = max(0, (duration - voiced_seconds) / duration) if duration > 0 else 0
        speaking_rate = num_segments / duration if duration > 0 else 0

        features = {
            'Duration': duration,
            'Pitch_Mean': pitch_mean,
            'Pitch_Std': pitch_std,
            'Jitter': jitter_local,
            'Shimmer': shimmer_local,
            'Speaking_Rate': speaking_rate,
            'Silence_Ratio': silence_ratio,
        }
        for idx, val in enumerate(mfcc_means):
            features[f'MFCC_{idx+1}'] = val
            
        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# --- Main Prediction Logic (Refactored to return a dictionary) ---

def predict_from_audio_path(file_path):
    """
    Takes a file path, runs the full prediction pipeline, and returns a result dictionary.
    """
    if not all([model, scaler, feature_names]):
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    # 1. Preprocess audio
    processed_path = preprocess_audio(file_path)
    if not processed_path:
        raise HTTPException(status_code=400, detail="Audio preprocessing failed.")

    # 2. Extract features
    features_dict = extract_features(processed_path)
    if not features_dict:
        os.remove(processed_path)
        raise HTTPException(status_code=400, detail="Feature extraction failed.")

    try:
        # 3. Convert to DataFrame and ensure correct column order
        feature_df = pd.DataFrame([features_dict])
        feature_df = feature_df[feature_names] # Crucial step!

        # 4. Scale features
        scaled_features = scaler.transform(feature_df)

        # 5. Make a prediction
        prediction_prob = model.predict(scaled_features, verbose=0)[0][0]
        prediction_label = int((prediction_prob > 0.5).astype("int32"))

        # 6. Format the result
        result_text = "Parkinson's Detected" if prediction_label == 1 else "Healthy"
        
        # 7. Cleanup the temporary processed file
        os.remove(processed_path)
        
        return {
            "status": "success",
            "prediction": result_text,
            "confidence": float(prediction_prob),
            "label": prediction_label
        }
    except Exception as e:
        # Ensure cleanup even if an error occurs after file creation
        os.remove(processed_path)
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


# --- FastAPI App Definition ---

app = FastAPI(
    title="Parkinson's Voice Detection API",
    description="An API that uses a deep learning model to predict the presence of Parkinson's disease from a voice recording.",
    version="1.0"
)

@app.get("/", tags=["General"])
def read_root():
    """A welcome message to check if the API is running."""
    return {"message": "Welcome to the Parkinson's Voice Prediction API. Go to /docs for usage."}

@app.post("/predict/", tags=["Prediction"])
async def create_prediction(file: UploadFile = File(...)):
    """
    Accepts an audio file, processes it, and returns the prediction result.
    The audio file can be in any format that librosa supports (wav, mp3, etc.).
    """
    # Save the uploaded file to a temporary location on the server
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling the uploaded file: {e}")

    # Now, run the prediction on the saved temporary file
    try:
        result = predict_from_audio_path(tmp_file_path)
        return JSONResponse(content=result)
    finally:
        # CRITICAL: Always clean up the temporary file
        os.remove(tmp_file_path)