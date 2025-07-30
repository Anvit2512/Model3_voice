import requests
import os

# --- CONFIGURATION ---
# Replace with your actual Hugging Face Space URL
API_URL = "https://parkinsons-api-8keh.onrender.com/predict" 

# Path to the audio file you want to test
AUDIO_FILE_PATH = "C:/Users/dell/3D Objects/summers/NeuroSentry AI/Model_3/backend/test2.unknown" 

# --- SCRIPT ---
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Error: Audio file not found at '{AUDIO_FILE_PATH}'")
else:
    print(f"🎙️ Sending audio file: {os.path.basename(AUDIO_FILE_PATH)} to API...")

    # The file needs to be opened in binary read mode 'rb'
    with open(AUDIO_FILE_PATH, 'rb') as audio_file:
        # The 'files' dictionary is where you specify the file to upload.
        # The key 'file' MUST match the name of the parameter in the FastAPI endpoint:
        # async def create_prediction(file: UploadFile = File(...)):
        files = {'file': (os.path.basename(AUDIO_FILE_PATH), audio_file, 'audio/wav')}
        
        try:
            response = requests.post(API_URL, files=files)
            
            # Raise an exception if the request was unsuccessful (e.g., 4xx or 5xx)
            response.raise_for_status() 
            
            # Get the JSON result from the response
            result = response.json()
            
            print("\n" + "="*30)
            print("      ✅ API RESPONSE RECEIVED")
            print("="*30)
            print(f" 🔹 Prediction: {result.get('prediction')}")
            print(f" 🔹 Confidence: {result.get('confidence'):.2%}")
            print("="*30)

        except requests.exceptions.RequestException as e:
            print(f"\n❌ An error occurred while contacting the API: {e}")
            if e.response:
                print(f"   Server responded with: {e.response.text}")