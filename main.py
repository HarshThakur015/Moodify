def detect_emotion(user_input, model_predict_fn):
    input_text = user_input.strip().lower()

    # Enhanced rule-based fallback for short or clear positive/relationship phrases
    love_keywords = [
        "love", "girlfriend", "boyfriend", "wife", "husband", "partner", "crush", "fiance", "fiancé", "fiancée",
        "in love", "my love", "my heart", "my soulmate", "my partner", "my girl", "my man", "my person"
    ]
    if len(input_text.split()) <= 2 or any(lk in input_text for lk in love_keywords):
        # Rule-based fallback for short or clear relationship/positive inputs
        if any(lk in input_text for lk in love_keywords):
            return "love", 1.0
        if "sad" in input_text:
            return "sadness", 1.0
        elif "angry" in input_text or "mad" in input_text:
            return "anger", 1.0
        elif "happy" in input_text or "joy" in input_text:
            return "joy", 1.0
        elif "fear" in input_text or "scared" in input_text:
            return "fear", 1.0
        elif "surprise" in input_text:
            return "surprise", 1.0

    # Use model prediction
    predicted_emotion, confidence = model_predict_fn(input_text)

    if confidence >= 0.6:
        return predicted_emotion, confidence
    else:
        return None, confidence
import random
import webbrowser
from yt_dlp import YoutubeDL
from mood_keywords import mood_keywords

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer once

_tokenizer = None
_model = None
_device = None
_labels = ["anger", "joy", "sadness", "fear", "surprise", "love"]

def load_emotion_model():
    global _tokenizer, _model, _device
    if _tokenizer is None or _model is None or _device is None:
        _tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        _model.to(_device)
    return _tokenizer, _model, _device

def predict_emotion(text):
    tokenizer, model, device = load_emotion_model()
    inputs = tokenizer(text, return_tensors="pt")
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return _labels[pred_idx], float(probs[pred_idx])

import streamlit as st

def get_random_keyword(emotion):
    # Use Streamlit session state to avoid repeating the last song for the same mood
    if 'last_song' not in st.session_state:
        st.session_state['last_song'] = {}
    songs = mood_keywords[emotion][:]
    last = st.session_state['last_song'].get(emotion)
    if last in songs and len(songs) > 1:
        songs.remove(last)
    song = random.choice(songs)
    st.session_state['last_song'][emotion] = song
    return song

def search_youtube_url(query):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': 'in_playlist',
        'default_search': 'ytsearch',
        'noplaylist': True,
        'forcejson': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(query, download=False)
        if 'entries' in result and result['entries']:
            video = result['entries'][0]
            return f"https://www.youtube.com/watch?v={video['id']}"
        return None

def recommend_song_for_text(text):
    emotion, score = detect_emotion(text, predict_emotion)
    if not emotion:
        return None, None, None, score
    keyword = get_random_keyword(emotion)
    url = search_youtube_url(keyword)
    if url:
        webbrowser.open(url)
    return emotion, keyword, url, score
