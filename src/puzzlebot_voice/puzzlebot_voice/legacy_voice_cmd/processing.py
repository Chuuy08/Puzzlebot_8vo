import numpy as np

""" 
    La normalización de la señal se realiza para eliminar variaciones en la amplitud debidas a condiciones de
    grabación, garantizando que las características extraídas dependan únicamente de la forma espectral de la
    señal y no de su intensidad
"""
# Normalize data
def normalize(signal):
    return (signal / np.max(np.abs(signal)))

"""
    El filtro de pre-énfasis se utiliza para compensar la caída espectral natural de la voz humana, la cual 
    presenta mayor energía en bajas frecuencias.
"""
# Pre emphasis filter
def pre_emphasis(signal, alpha=0.95):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

"""
    El proceso de framing consiste en dividir la señal de audio en segmentos más pequeños, llamados frames, que
    son lo suficientemente cortos para suponer que la señal es estacionaria dentro de cada frame. 

"""
# Framing
# def framing(signal, frame_length=320, hop_length=128):
#     num_frames = int(np.floor((len(signal) - frame_length) / hop_length))
#     frames = []
#     for i in range(num_frames):
#         start = i * hop_length
#         frames.append(signal[start:start + frame_length])
#     return np.array(frames)

def framing(signal, fs):
        frame_length = 320 # fixed to 320 points as requested
        hop_length = 128   # fixed to 128 samples as requested
        
        num_frames = 1 + (len(signal) - frame_length) // hop_length

        frames = np.lib.stride_tricks.as_strided(
            signal, 
            shape=(num_frames, frame_length), 
            strides=(signal.strides[0] * hop_length, signal.strides[0])
        )
        return np.array(frames)

"""
    La ventana de Hamming se utiliza para suavizar los extremos de cada frame, reduciendo las discontinuidades 
    introducidas por el truncamiento de la señal. Esto mejora la estimación espectral y evita efectos no deseados 
    en el cálculo de características como la autocorrelación y los coeficientes LPC
"""

# Hamming window
def hamming_window(frames):
    frame_length = frames.shape[1]
    window = np.hamming(frame_length)
    return frames * window[np.newaxis, :]


"""
    Esta parte nos permite eliminar segmentos de silencio y ruidos presentes en la se;al original, conservando
    unicamente las partes que continen la porsion donde existe actividad de voz.
"""
# Deteccion inicio y fin (energia + ZCR)
def detect_voice(signal, fs):
    frame_length = 320 # fixed to 320 points
    hop_length = 128   # fixed to 128 points

    num_frames = 1 + (len(signal) - frame_length) // hop_length

    zcr = []
    energy = []

    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]

        crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr.append(crossings / frame_length)

        energy.append(np.sum(frame**2) / frame_length)

    zcr = np.array(zcr)
    energy = np.array(energy)

    zcr_th = 0.08 * np.max(zcr)
    energy_th = 0.03 * np.max(energy)

    voice = (zcr > zcr_th) & (energy > energy_th)

    idx = np.where(voice)[0]
    if len(idx) == 0:
        return signal

    start = idx[0] * hop_length
    end = idx[-1] * hop_length + frame_length

    return signal[start:end]
