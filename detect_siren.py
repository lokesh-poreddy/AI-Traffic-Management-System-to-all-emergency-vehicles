import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Bandpass filter to isolate siren frequency range (400Hz–1600Hz)
def bandpass_filter(signal, sr, lowcut=400, highcut=1600, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Function to detect siren sound
def detect_siren(audio_path, threshold=0.6, plot_spectrogram=False):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=22050)

        # Apply bandpass filter
        y_filtered = bandpass_filter(y, sr)

        # Compute Log-Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=40, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Compute mean energy in the siren-dominant frequency range
        siren_score = np.mean(log_mel_spec[5:15, :])  # Focus on bands between ~400Hz-1600Hz

        # Normalize score (Min-Max Scaling)
        normalized_score = (siren_score - np.min(log_mel_spec)) / (np.max(log_mel_spec) - np.min(log_mel_spec))

        # Plot spectrogram (optional)
        if plot_spectrogram:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Log-Mel Spectrogram")
            plt.show()

        # Determine if it's a siren
        is_siren = normalized_score > threshold
        return {"siren_detected": is_siren, "confidence": round(normalized_score, 2)}

    except Exception as e:
        return {"error": str(e)}

# Example Usage
if __name__ == "__main__":
    audio_file = "/Users/meghanadhkottana/Documents/pythonProjects/siren.mp3"

    if os.path.exists(audio_file):
        result = detect_siren(audio_file, plot_spectrogram=True)  # Set to False if no visualization needed
        print(result)
    else:
        print("Audio file not found.")
