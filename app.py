import streamlit as st

st.title("Audio Processing Demo")

st.write("""
This is a basic Streamlit app that demonstrates the use of audio processing libraries.
You can upload an audio file and perform basic operations on it.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    from model_demo import run_denoising_streamlit
    run_denoising_streamlit(st, "temp_audio.wav") 