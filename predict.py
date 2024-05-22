import tensorflow as tf
import tensorflow_io as tfio  # Ensure this import for audio processing
import matplotlib.pyplot as plt

import sys


# Function to load and preprocess wav file to 16kHz mono
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

def classify_chirp(file_path):
    with open(file_path, "rb") as f:
        waveform = load_wav_16k_mono(f)
    chirps_results = chirpify_model(waveform[tf.newaxis, :])
    top_class = tf.math.argmax(chirps_results, axis=-1).numpy()[0]
    inferred_class = cricket_classes[top_class]
    class_probabilities = tf.nn.softmax(chirps_results, axis=-1).numpy()
    top_score = class_probabilities[0, top_class]
    return inferred_class, top_score

filename = r"D:\Cricket Thesis\Recordings\Audio Dataset Final\MRPS-1_REC-5142024-0034_4.wav"


with open(filename, "rb") as f:
    waveform = load_wav_16k_mono(f)

chirps_results = chirpify_model(waveform)
top_class= tf.math.argmax(chirps_results)
inferred_class = cricket_classes[top_class]
class_probabilities = tf.nn.softmax(chirps_results, axis=-1)
top_score = class_probabilities[top_class]
print(f'[Chirpify] The main sound is: {inferred_class} ({top_score})')


plt.figure(figsize=(10, 6))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])

# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.
mean_scores = np.mean(scores, axis=0)
top_n = 8
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [cricket_classes[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([top_n, 0]))

