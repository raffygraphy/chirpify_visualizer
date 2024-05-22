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

def segment_audio(waveform, segment_length=2, sample_rate=16000):
    segment_samples = segment_length * sample_rate
    num_segments = len(waveform) // segment_samples
    segments = [waveform[i*segment_samples:(i+1)*segment_samples] for i in range(num_segments)]
    return segments

def classify_segments(segments, model):
    predictions = []
    for segment in segments:
        segment = tf.convert_to_tensor(segment, dtype=tf.float32)
        inputs = {'audio': segment}
        my_scores = model(**inputs)['classifier']
        my_scores_np = my_scores.numpy()
        my_top_class = tf.math.argmax(my_scores_np)
        predictions.append(my_top_class.numpy())
        
        class_probabilities = tf.nn.softmax(my_scores, axis=-1)
        top_score = class_probabilities[my_top_class].numpy() * 100

        top_score_str = f"{top_score:.10f}"[:6]  # Ensuring we get at least 2 decimal places
        print(f'[Chirpify] Class Probability ({top_score_str})')
    return predictions

def plot_waveform_with_segments_and_heatmaps(waveform, sample_rate, segment_lengths, all_predictions, classes):
    time = np.arange(len(waveform)) / sample_rate
    num_plots = 1 + len(segment_lengths) * 2  
    
    # Create a figure with specified size and facecolor
    fig = plt.figure(figsize=(15, 2 * num_plots), facecolor='none')
    
    # Create subplots
    axes = fig.subplots(num_plots, 1, sharex=True)
    
    # Plot original waveform
    axes[0].plot(time, waveform, color='orange')
    axes[0].set_title('Original Waveform', color='white')
    axes[0].set_ylabel('Amplitude', color='white')
    axes[0].tick_params(axis='both', colors='white')
    axes[0].grid(True, color='gray')
    axes[0].set_facecolor('none')  # Set transparent background

    
    # Plot segmentations and heatmaps
    for i, (segment_length, predictions) in enumerate(zip(segment_lengths, all_predictions)):
        segment_index = 1 + i * 2
        heatmap_index = 2 + i * 2
        
        # Plot segmentations
        axes[segment_index].plot(time, waveform, color='orange')
        num_segments = len(waveform) // (segment_length * sample_rate)
        for j in range(num_segments):
            axes[segment_index].axvline(j * segment_length, color='r', linestyle='--')
        axes[segment_index].set_title(f'Segment Length: {segment_length} seconds', color='white')
        axes[segment_index].set_ylabel('Amplitude', color='white')
        axes[segment_index].tick_params(axis='both', colors='white')
        axes[segment_index].grid(True, color='gray')
        axes[segment_index].set_facecolor('none')  # Set transparent background
        
        # Plot heatmap
        num_segments = len(predictions[0])
        axes[heatmap_index].imshow(predictions, cmap='cividis', aspect='auto', interpolation='nearest', extent=[0, num_segments * segment_length, 0, len(classes)])
        axes[heatmap_index].set_title(f'Inferred Classes Heatmap (Segment Length: {segment_length} seconds)', color='white')
        axes[heatmap_index].set_xlabel('Time (seconds)', color='white')
        axes[heatmap_index].set_ylabel('Classes', color='white')
        axes[heatmap_index].tick_params(axis='both', colors='white')
        axes[heatmap_index].set_yticks(np.arange(0.5, len(classes) + 0.5, 1))
        axes[heatmap_index].set_yticklabels(classes[::-1])
    
    # Adjust layout
    fig.tight_layout()
    
    # Show plot in Streamlit
    st.pyplot(fig)

# Streamlit app layout
st.title('Chirpify: Insect Audio Classifier [Orthoptera Edition]')
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.write("Uploaded file details:")
    st.write(f"File name: {uploaded_file.name}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    samplerate, data = wavfile.read(temp_file_path)

    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data - 128) / 128.0


    st.markdown('---')
    st.subheader("Audio Playback")

    audio_file = open(temp_file_path, "rb")

    st.audio(audio_file.read())

    st.markdown('---')
    
    st.subheader("Waveform")

    plt.figure(figsize=(10, 4), facecolor='none')
    plt.plot(data, color='orange')
    plt.ylim(-1, 1)
    plt.title('Normalized Audio Waveform', color='white')
    plt.xlabel('Sample Index', color='white')
    plt.ylabel('Amplitude', color='white')
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('none')
    plt.tick_params(colors='white')
    plt.yticks(color='white')
    plt.xticks(color='white')
    st.pyplot(plt)

    filename = temp_file_path

    with open(filename, "rb") as f:
        waveform = load_wav_16k_mono(f)

    model_path = r"D:\Cricket Thesis\cricket_yamnet"
    reloaded_model = tf.saved_model.load(model_path)
    model = reloaded_model.signatures["serving_default"]



    my_classes = [
        'Chorthippus biguttulus',
        'Chorthippus brunneus',
        'MRPS-1',
        'MRPS-2',
        'Oecanthus pellucens',
        'Pholidoptera griseoaptera',
        'Pseudochorthippus parallelus',
        'Tettigonia viridissima'
    ]

    chirps_results = reloaded_model(waveform)
    top_class= tf.math.argmax(chirps_results)
    inferred_class = my_classes[top_class]
    class_probabilities = tf.nn.softmax(chirps_results, axis=-1)
    top_score = class_probabilities[top_class].numpy() * 100

   

    top_score_str = f"{top_score:.10f}"[:6]  # Ensuring we get at least 2 decimal places

    print(f'[Chirpify] The main sound is: {inferred_class} ({top_score})')

    st.subheader("Overall Classification Results")

    #  # Display the inferred class and top score in Streamlit
    # st.markdown(f'<h2>Inferred Class: {inferred_class}</h2>', unsafe_allow_html=True)
    # st.markdown(f'<h2>Confidence Score: {top_score_str}%</h2>', unsafe_allow_html=True)

    st.markdown(f'''
    Inferred Class: <span style="color: orange;">{inferred_class}</span>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
    Confidence Score: <span style="color: orange;">{top_score}</span>
    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.subheader("Segmentated Classification Results")

    segment_lengths = [1, 2, 3, 4]

    all_predictions = []
    for segment_length in segment_lengths:
        segments = segment_audio(waveform, segment_length=segment_length, sample_rate=16000)
        predictions = classify_segments(segments, model)
        predictions_matrix = np.zeros((len(my_classes), len(predictions)))
        for j, pred_class_idx in enumerate(predictions):
            predictions_matrix[pred_class_idx, j] = 1
        all_predictions.append(predictions_matrix)
        for i, pred_class_idx in enumerate(predictions):
            inferred_class = my_classes[pred_class_idx]
            # st.write(f"Segment {i} (seconds {i*segment_length} to {(i+1)*segment_length}): {inferred_class}")

    plot_waveform_with_segments_and_heatmaps(waveform, sample_rate=16000, segment_lengths=segment_lengths, all_predictions=all_predictions, classes=my_classes)
