import os
import numpy as np
import scipy.io.wavfile as wav

def load_dataset(path):
    data = {}
    for word in os.listdir(path):
        word_path = os.path.join(path, word)

        if not os.path.isdir(word_path):
            continue

        signals = []
        for file in sorted(os.listdir(word_path)):
            file_path = os.path.join(word_path, file)

            if not file.endswith(".wav"):
                continue

            fs, signal = wav.read(file_path)
            if signal.ndim > 1:
                signal = signal[:, 0]  
            signal = signal.astype(float) / 32768.0
            signals.append((fs, signal))

        data[word] = signals

    return data