# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.io import wavfile
# import tempfile

# # Streamlit app layout
# st.title('Chirpify: Chirp Audio Classifier: Orthoptera Edition')
# uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# # Load the model
# model = load_model()

# if uploaded_file is not None:
#     # Print the file details
#     st.write("Uploaded file details:")
#     st.write(f"File name: {uploaded_file.name}")
    
#     # Create a temporary file and write the uploaded content to it
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         temp_file_path = tmp_file.name

#     # Read the audio file
#     samplerate, data = wavfile.read(temp_file_path)

#     # Normalize the audio data to the range -1 to 1
#     if data.dtype == np.int16:
#         data = data / 32768.0
#     elif data.dtype == np.int32:
#         data = data / 2147483648.0
#     elif data.dtype == np.uint8:
#         data = (data - 128) / 128.0

#     # Plot the normalized audio data
#     plt.figure(figsize=(10, 4))
#     plt.plot(data)
#     plt.ylim(-1, 1)
#     plt.title('Normalized Audio Waveform')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Amplitude')
#     plt.grid(True)

#     # Display the plot in Streamlit
#     st.pyplot(plt)

#     # Print the temporary file path
#     st.write(f"Temporary file path: {temp_file_path}")



# # Function to load and preprocess wav file to 16kHz mono
# def load_wav_16k_mono(file):
#     file_contents = file.read()
#     wav, sample_rate = tf.audio.decode_wav(
#         file_contents,
#         desired_channels=1
#     )
#     wav = tf.squeeze(wav, axis=-1)
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
#     return wav

# cricket_classes = [
#     'Chorthippus biguttulus',
#     'Chorthippus brunneus',
#     'Oecanthus pellucens',
#     'Pholidoptera griseoaptera',
#     'Pseudochorthippus parallelus',
#     'Tettigonia viridissima'
# ]

# model_path = r"D:\Cricket Thesis\cricket_yamnet"
# chirpify_model = tf.saved_model.load(model_path)

# def classify_chirp(file_path):
#     with open(file_path, "rb") as f:
#         waveform = load_wav_16k_mono(f)
#     chirps_results = chirpify_model(waveform[tf.newaxis, :])
#     top_class = tf.math.argmax(chirps_results, axis=-1).numpy()[0]
#     inferred_class = cricket_classes[top_class]
#     class_probabilities = tf.nn.softmax(chirps_results, axis=-1).numpy()
#     top_score = class_probabilities[0, top_class]
#     return inferred_class, top_score


# def load_model():
#     model_path = r"D:\Cricket Thesis\cricket_yamnet"
#     chirpify_model = tf.saved_model.load(model_path)

#     return model

# # Load the model
# model = load_model()


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import tempfile
import tensorflow as tf
import tensorflow_io as tfio

# Function to load the model
@st.cache_data(allow_output_mutation=True)
def load_model():
    model_path = r"D:\Cricket Thesis\chirpify\cricket_yamnet"
    chirpify_model = tf.saved_model.load(model_path)
    return chirpify_model

# Function to load and preprocess wav file to 16kHz mono
def load_wav_16k_mono(file):
    file_contents = file.read()
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Function to classify chirp audio
def classify_chirp(file_path, model):
    with open(file_path, "rb") as f:
        waveform = load_wav_16k_mono(f)
    chirps_results = model(waveform[tf.newaxis, :])
    top_class = tf.math.argmax(chirps_results, axis=-1).numpy()[0]
    inferred_class = cricket_classes[top_class]
    class_probabilities = tf.nn.softmax(chirps_results, axis=-1).numpy()
    top_score = class_probabilities[0, top_class]
    return inferred_class, top_score

# Define the cricket classes
cricket_classes = [
    'Chorthippus biguttulus',
    'Chorthippus brunneus',
    'Oecanthus pellucens',
    'Pholidoptera griseoaptera',
    'Pseudochorthippus parallelus',
    'Tettigonia viridissima'
]

# Load the model
model = load_model()

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

    # Plot the normalized audio data
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.ylim(-1, 1)
    plt.title('Normalized Audio Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Classify the chirp audio
    inferred_class, top_score = classify_chirp(temp_file_path, model)

    # Print the classification result
    st.write(f"Predicted Class: {inferred_class}")
    st.write(f"Confidence Score: {top_score:.2f}")

    # Print the temporary file path (for debugging purposes)
    st.write(f"Temporary file path: {temp_file_path}")
