import streamlit as st
from main import recommend_song_for_text

st.set_page_config(page_title="Emotion-Based Song Recommender", page_icon="🎵", layout="centered")

st.markdown("""
<h1 style='text-align: center; color: #4F8BF9;'>🎵 Emotion-Based Song Recommender</h1>
<p style='text-align: center; font-size: 1.2em;'>
Tell me how you feel, and I'll recommend a song that matches your mood!
</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stTextInput>div>div>input {
    font-size: 1.1em;
    padding: 0.5em;
}
.song-box {
    background: transparent;
    border-radius: 0;
    padding: 0;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    user_input = st.text_input("How are you feeling today?", placeholder="e.g. I'm feeling happy and energetic!")
    btn = st.button("🔍 Detect Emotion & Recommend Song", use_container_width=True)

if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

if btn:
    if not user_input.strip():
        st.warning("Please enter a sentence about how you feel.")
        st.session_state['last_result'] = None
    else:
        import time
        progress_text = "Loading model and detecting emotion..."
        progress_bar = st.progress(0, text=progress_text)
        for percent in range(0, 80, 8):
            time.sleep(0.05)
            progress_bar.progress(percent, text=progress_text)
        # Actual prediction
        emotion, keyword, url, score = recommend_song_for_text(user_input)
        for percent in range(80, 101, 4):
            time.sleep(0.01)
            progress_bar.progress(percent, text=progress_text)
        progress_bar.empty()
        if not emotion:
            st.session_state['last_result'] = {
                'error': "Sorry, I couldn't confidently detect your emotion. Try describing it in more detail."
            }
        else:
            st.session_state['last_result'] = {
                'emotion': emotion,
                'keyword': keyword,
                'url': url,
                'score': score
            }

if st.session_state['last_result']:
    res = st.session_state['last_result']
    if 'error' in res:
        st.error(res['error'])
    else:
        emoji_map = {
            'joy': '😄',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲',
            'love': '❤️'
        }
        emj = emoji_map.get(res['emotion'], '🎵')
        st.markdown("<div class='song-box'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:1.1em; margin-bottom:0.5em;'><b>{emj} {res['emotion'].capitalize()} (confidence: {res['score']:.2f})</b></div>",
            unsafe_allow_html=True
        )
        if res['url'] and res['keyword']:
            st.markdown(
                f"<b>Recommended Song:</b> <a href='{res['url']}' target='_blank'>{res['keyword']}</a>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<b>Sorry, couldn't find a song for this emotion. Try again!</b>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
