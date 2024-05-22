import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import tempfile
import tensorflow as tf
import tensorflow_io as tfio


def load_wav_16k_mono(file):
    file_contents = file.read()
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1
    )
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

cricket_classes = [
    'Chorthippus biguttulus',
    'Chorthippus brunneus',
    'MRPS-1',
    'MRPS-2',
    'Oecanthus pellucens',
    'Pholidoptera griseoaptera',
    'Pseudochorthippus parallelus',
    'Tettigonia viridissima'
]

model_path = r"D:\Cricket Thesis\cricket_yamnet"
chirpify_model = tf.saved_model.load(model_path)

# Streamlit app layout
st.title('Chirpify: Chirp Audio Classifier: Orthoptera Edition')
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Print the file details
    st.write("Uploaded file details:")
    st.write(f"File name: {uploaded_file.name}")
    
    # Create a temporary file and write the uploaded content to it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    # Read the audio file
    samplerate, data = wavfile.read(temp_file_path)

    # Normalize the audio data to the range -1 to 1
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data - 128) / 128.0


        # Plot the normalized audio data in night mode
    plt.figure(figsize=(10, 4), facecolor='none')  # Set background to transparent
    plt.plot(data, color='orange')
    plt.ylim(-1, 1)
    plt.title('Normalized Audio Waveform', color='white')
    plt.xlabel('Sample Index', color='white')
    plt.ylabel('Amplitude', color='white')
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('none')  # Set axes background to transparent
    plt.tick_params(colors='white')
    plt.yticks(color='white')
    plt.xticks(color='white')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Print the temporary file path
    # st.write(f"Temporary file path: {temp_file_path}")
    filename = temp_file_path

    with open(filename, "rb") as f:
     waveform = load_wav_16k_mono(f)

    chirps_results = chirpify_model(waveform)
    top_class= tf.math.argmax(chirps_results)
    inferred_class = cricket_classes[top_class]
    class_probabilities = tf.nn.softmax(chirps_results, axis=-1)
    top_score = class_probabilities[top_class].numpy() * 100
    # print(f'[Chirpify] The main sound is: {inferred_class} ({top_score})')

    top_score_str = f"{top_score:.10f}"[:6]  # Ensuring we get at least 2 decimal places

    print(f'[Chirpify] The main sound is: {inferred_class} ({top_score})')

     # Display the inferred class and top score in Streamlit
    st.markdown(f'<h2>Inferred Class: {inferred_class}</h2>', unsafe_allow_html=True)
    st.markdown(f'<h2>Confidence Score: {top_score_str}%</h2>', unsafe_allow_html=True)





