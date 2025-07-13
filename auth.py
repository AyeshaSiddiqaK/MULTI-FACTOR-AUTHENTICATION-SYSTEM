import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import cv2
import numpy as np
import os
import face_recognition
import sounddevice as sd
import soundfile as sf
import librosa
import whisper
from fuzzywuzzy import fuzz
import noisereduce as nr
import time
import random
import mediapipe as mp  


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


FACE_DATA_PATH = "face_data"
VOICE_DATA_PATH = "voice_samples"
VIDEO_DATA_PATH = "video_samples"
os.makedirs(FACE_DATA_PATH, exist_ok=True)
os.makedirs(VOICE_DATA_PATH, exist_ok=True)
os.makedirs(VIDEO_DATA_PATH, exist_ok=True)


TEXT_PROMPTS = [
    "Hello, how are you?", "I love pizza.", "The sky is blue.", "I like ice cream.", "My name is Alex.",
    "Today is a good day.", "I enjoy reading.", "The cat is cute.", "I like to dance.", "The sun is bright.",
    "I love my dog.", "The weather is nice.", "I like chocolate.", "I enjoy swimming.", "The moon is beautiful.",
    "I love my family.", "I like to travel.", "The flowers are pretty.", "I enjoy cooking.", "The stars are shining.",
    "I love my job.", "I like to sing.", "The ocean is vast.", "I enjoy hiking.", "The birds are chirping.",
    "I love my friends.", "I like to draw.", "The mountains are tall.", "I enjoy gardening.", "The wind is blowing.",
    "I love my home.", "I like to write.", "The river is flowing.", "I enjoy cycling.", "The clouds are fluffy.",
    "I love my car.", "I like to play games.", "The forest is peaceful.", "I enjoy photography.", "The rain is falling.",
    "I love my phone.", "I like to watch movies.", "The desert is hot.", "I enjoy painting.", "The snow is cold.",
    "I love my bike.", "I like to listen to music.", "The beach is sandy.", "I enjoy jogging.", "The park is green."
]


def capture_face_image():
    cap = cv2.VideoCapture(0)
    time.sleep(1)  
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def record_video_and_audio(duration=5):
    print("🔴 Recording video and audio... Speak clearly.")
    cap = cv2.VideoCapture(0)
    fs = 44100  
    video_frames = []
    audio_frames = []

    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
        
        audio_frame = sd.rec(int(0.1 * fs), samplerate=fs, channels=1, blocking=True)
        audio_frames.append(audio_frame.flatten())

    cap.release()
    sd.stop()

    
    audio_frames = np.concatenate(audio_frames)
    cleaned_audio = nr.reduce_noise(y=audio_frames, sr=fs)
    audio_file = "temp_audio.wav"
    sf.write(audio_file, cleaned_audio, fs)
    print("✅ Video and audio saved.")
    return video_frames, audio_file


def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return np.mean(mfcc, axis=1)


def detect_lip_movements(frame):
    
    height, width, _ = frame.shape
    if height != width:
        size = min(height, width)
        frame = frame[:size, :size]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None
    
    
    lip_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        for idx in list(range(61, 68)) + list(range(291, 308)):
            lip_landmarks.append((face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y))
    return lip_landmarks


def check_lip_sync(video_frames, audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    spoken_text = result["text"].strip().lower()

    lip_movement_count = 0
    total_frames = len(video_frames)
    
    for frame in video_frames:
        lip_landmarks = detect_lip_movements(frame)
        if lip_landmarks is not None:
            lip_movement_count += 1

    # Set a stricter threshold: at least 50% of frames must have lip movement
    if total_frames > 0 and (lip_movement_count / total_frames) > 0.7:
        return True, "Lip-sync verified."
    else:
        return False, "Lip-sync check failed! Not enough consistent lip movements detected."



def record_voice(filename, duration=5, fs=44100):
    print(f"🎙️ Recording voice for {duration} seconds... Speak clearly.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  
    reduced_audio = nr.reduce_noise(y=audio.flatten(), sr=fs)  
    sf.write(filename, reduced_audio, fs)
    print(f"✅ Voice sample saved as {filename}")



def register_user():
    username = input("Enter username: ").strip().lower()
    face_file = os.path.join(FACE_DATA_PATH, f"{username}.npy")
    voice_dir = os.path.join(VOICE_DATA_PATH, username)
    
    if os.path.exists(face_file) or os.path.exists(voice_dir):
        print("❌ Username already exists! Choose a different name.")
        return
    
    
    print("📸 Capturing face for registration...")
    frame = capture_face_image()
    if frame is None:
        print("❌ Failed to capture face image!")
        return
    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        print("❌ No face detected! Try again.")
        return
    np.save(face_file, face_encodings[0])
    
    
    os.makedirs(voice_dir, exist_ok=True)
    for i in range(3):  
        prompt = random.choice(TEXT_PROMPTS)
        print(f"📢 Please read this aloud for sample {i+1}: {prompt}")
        voice_file = os.path.join(voice_dir, f"{username}_sample_{i+1}.wav")
        record_voice(voice_file)
    
    print(f"✅ User '{username}' registered successfully!")


def authenticate_user():
    print("🔍 Capturing face and voice for authentication...")
    
    
    prompt = random.choice(TEXT_PROMPTS)
    print(f"📢 Please read this aloud: {prompt}")
    video_frames, audio_file = record_video_and_audio()
    
    
    known_faces = {f.split('.')[0]: np.load(os.path.join(FACE_DATA_PATH, f)) for f in os.listdir(FACE_DATA_PATH) if f.endswith(".npy")}
    if not known_faces:
        print("❌ No registered users found! Please register first.")
        return
    
    
    current_face_encoding = face_recognition.face_encodings(video_frames[-1])
    if not current_face_encoding:
        print("❌ No face detected in the video!")
        return
    
    
    face_matches = face_recognition.compare_faces(list(known_faces.values()), current_face_encoding[0], tolerance=0.5)
    if not any(face_matches):
        print("❌ Face not recognized!")
        return
    
    
    matched_username = list(known_faces.keys())[face_matches.index(True)]
    print(f"✅ Face verified. Recognized as {matched_username.capitalize()}.")
    
    
    voice_samples = [os.path.join(VOICE_DATA_PATH, matched_username, f) for f in os.listdir(os.path.join(VOICE_DATA_PATH, matched_username)) if f.endswith(".wav")]
    if not voice_samples:
        print("❌ No voice samples found for this user!")
        return
    
    test_features = extract_voice_features(audio_file)
    similarities = []
    for sample in voice_samples:
        sample_features = extract_voice_features(sample)
        similarity = np.linalg.norm(test_features - sample_features)
        similarities.append(similarity)
    avg_similarity = np.mean(similarities)
    if avg_similarity > 0.5:  
        print("❌ Voice not recognized!")
        return
    print("✅ Voice verified.")
    
    
    lip_sync_result, lip_sync_message = check_lip_sync(video_frames, audio_file)
    if not lip_sync_result:
        print(f"❌ {lip_sync_message}")
        return
    print(f"✅ {lip_sync_message}")
    
    
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    spoken_text = result["text"].strip().lower()
    similarity = fuzz.ratio(prompt.lower(), spoken_text.lower())
    if similarity > 95:  
        print("❌ Incorrect text spoken!")
        return
    print("✅ Spoken text verified.")
    
    
    print(f"✅ Access granted! Welcome, {matched_username.capitalize()}.")


if __name__ == "__main__":
    while True:
        print("\n=== Multi-Factor Authentication System ===")
        print("1️⃣ Register a New User")
        print("2️⃣ Authenticate")
        print("3️⃣ Exit")
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            register_user()
        elif choice == '2':
            authenticate_user()
        elif choice == '3':
            print("👋 Exiting. Have a great day!")
            break
        else:
            print("❌ Invalid choice! Try again.")
