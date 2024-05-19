# import tensorflow as tf
# import tensorflow_io as tfio  # Ensure this import for audio processing
# import sys


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


# # if len(sys.argv) != 2:
# #         print("Usage: python predict.py <file_path>")
# #         sys.exit(1)

# # file_path = sys.argv[1]
# # inferred_class, top_score = classify_chirp(file_path)
# # print(f"Class: {inferred_class}, Score: {top_score}")

# # filename = r"D:\Cricket Thesis\chirpify\test-Oecanthus pellucens.wav"

# # with open(filename, "rb") as f:
# #     waveform = load_wav_16k_mono(f)

# # chirps_results = chirpify_model(waveform)
# # top_class= tf.math.argmax(chirps_results)
# # inferred_class = cricket_classes[top_class]
# # class_probabilities = tf.nn.softmax(chirps_results, axis=-1)
# # top_score = class_probabilities[top_class]
# # print(f'[Chirpify] The main sound is: {inferred_class} ({top_score})')

def classify_wav(wav_file):
    """Classifies a cricket sound from a WAV file."""

    # Read the WAV file
    waveform, sample_rate = tf.audio.decode_wav(wav_file.read(), desired_channels=1)

    # Resample to 16 kHz if necessary (check model requirements)
    if sample_rate != 16000:
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)

    # Squeeze the channel dimension if needed (model might expect mono)
    waveform = tf.squeeze(waveform, axis=-1)

    # Preprocess the waveform (if required by your model)
    # Example (replace with your model's preprocessing steps):
    # waveform = tf.cast(waveform, tf.float32)  # Normalize if needed

    # Pass the preprocessed waveform to the loaded model
    chirps_results = chirpify_model(waveform[tf.newaxis, :])

    # Get the top prediction class and score
    top_class = tf.math.argmax(chirps_results, axis=-1).numpy()[0]
    inferred_class = cricket_classes[top_class]
    top_score = tf.nn.softmax(chirps_results, axis=-1).numpy()[0, top_class]

    return inferred_class, top_score
