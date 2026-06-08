import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
from .processing import *
from .features import autocorrelation, extract_lpc, lpc_to_lsf
from .vq import *
from .utils import *
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
CODEBOOK_SIZES = [16, 32, 64]
P = 12  # orden LPC


def main():
    data = load_dataset(DATASET_PATH)
    labels = list(data.keys())

    for CODEBOOK_SIZE in CODEBOOK_SIZES:
        print(f"\nProbando con codebook de tamaño {CODEBOOK_SIZE}")

        codebooks = {}

        #  ENTRENAMIENTO

        for word in labels:
            features_all_P = []
            features_all_Q = []
            autocorr_all = []

            for i in range(10):  # 10 audios entrenamiento
                fs, signal = data[word][i]

                # Preprocesamiento
                signal = normalize(signal)
                signal = detect_voice(signal, fs)
                if len(signal) < 320:
                    continue
                signal = pre_emphasis(signal)

                frames = framing(signal, fs)
                if len(frames) == 0:
                    continue

                frames = hamming_window(frames)

                # LPC → LSF
                lpc, _ = extract_lpc(frames, P)
                lsf = np.array([lpc_to_lsf(l) for l in lpc])

                # Separar LSF_P y LSF_Q
                lsf_P = lsf[:, :P//2]
                lsf_Q = lsf[:, P//2:]

                # Autocorrelaciones
                autocorrs = np.array([autocorrelation(frame, P) for frame in frames])

                features_all_P.extend(lsf_P)
                features_all_Q.extend(lsf_Q)
                autocorr_all.extend(autocorrs)

            if len(features_all_P) > 0:
                codebooks[word] = lbg(np.array(features_all_P), np.array(features_all_Q), np.array(autocorr_all), CODEBOOK_SIZE)

        #  PRUEBA

        predictions = []
        true_labels = []

        for word in labels:
            for i in range(10, 15):  # 5 audios prueba
                fs, signal = data[word][i]

                signal = normalize(signal)
                signal = detect_voice(signal, fs)
                if len(signal) < 320:
                    continue
                signal = pre_emphasis(signal)

                frames = framing(signal, fs)
                if len(frames) == 0:
                    continue

                frames = hamming_window(frames)

                # Extraer autocorrelaciones
                autocorrs = []
                for frame in frames:
                    r = autocorrelation(frame, P)
                    autocorrs.append(r)

                autocorrs = np.array(autocorrs)
                predicted = recognize(autocorrs, codebooks, labels)
                predictions.append(predicted)
                true_labels.append(word)

        # 🔹 RESULTADOS

        cm = confusion_matrix(true_labels, predictions, labels=labels)
        accuracy = np.trace(cm) / np.sum(cm)

        print("Matriz de confusión:")
        print(cm)
        print(f"Precisión: {accuracy:.2f}")


if __name__ == '__main__':
    main()
