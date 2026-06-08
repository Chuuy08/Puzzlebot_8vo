import numpy as np
import scipy
from scipy.fftpack import dct
#import lisbrosa

class VoiceUtils():
    """ 
        La normalización de la señal se realiza para eliminar variaciones en la amplitud debidas a condiciones de
        grabación, garantizando que las características extraídas dependan únicamente de la forma espectral de la
        señal y no de su intensidad
    """
    def normalize(self,signal):
        return (signal / np.max(np.abs(signal)))
    """
        El filtro de pre-énfasis se utiliza para compensar la caída espectral natural de la voz humana, la cual 
        presenta mayor energía en bajas frecuencias.
    """
    def pre_emphasis(self,signal, alpha):
        return np.append(signal[0], signal[1:] - alpha * signal[:-1]) # y[n] = x[n] - alpha * x[n-1]
    """
        El proceso de framing consiste en dividir la señal de audio en segmentos más pequeños, llamados frames, que
        son lo suficientemente cortos para suponer que la señal es estacionaria dentro de cada frame. 

    """
    # TODO
    # - Change to librosa or adjust to numpy stride 
    def framing(self,signal, fs):
        frame_length = int(0.02 * fs) # changed from constant to variable
        hop_length = int(0.01 * fs)
        
        #num_frames = int(np.floor((len(signal) - frame_length) / hop_length)) #? 
        #frames = []

        #for i in range(num_frames):
        #    start = i * hop_length
        #    frames.append(signal[start:start + frame_length])
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
    def hamming_window(self,frames):
        frame_length = frames.shape[1]
        window = np.hamming(frame_length)
        return frames * window[np.newaxis, :]

    """
        Esta parte nos permite eliminar segmentos de silencio y ruidos presentes en la se;al original, conservando
        unicamente las partes que continen la porsion donde existe actividad de voz.
    """
    def detect_voice(self,signal, fs):
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

    
    def autocorrelation(self,frame, p):
        r = np.zeros(p + 1)
        for k in range(p + 1):
            r[k] = np.sum(frame[:len(frame) - k] * frame[k:])
        return r
    def levinson_durbin(self,r, p):
        a = np.zeros(p)
        e = r[0]
        for i in range(1, p + 1):
            acc = np.dot(a[:i-1], r[1:i][::-1])
            k = (r[i] - acc) / e
            a_new = a.copy()
            a_new[i - 1] = k
            for j in range(i - 1):
                a_new[j] = a[j] - k * a[i - j - 2]
            a = a_new
            e *= (1 - k * k)
        return np.concatenate(([1], -a)), e

    def extract_lpc(self,frames, order=12):
        lpc_vectors, sigmas = [], []
        for frame in frames:
            r = self.autocorrelation(frame, order)
            a, s = self.levinson_durbin(r, order)
            lpc_vectors.append(a)
            sigmas.append(s)
        return np.array(lpc_vectors), np.array(sigmas)

    def lpc_to_lsf(self,a):
        p = len(a) - 1
        a_pad = np.append(a, 0.0)
        a_rev = a_pad[::-1]
        
        P_poly = a_pad + a_rev
        Q_poly = a_pad - a_rev
        
        rP = np.roots(P_poly)
        rQ = np.roots(Q_poly)
        
        anglesP = np.angle(rP)
        anglesQ = np.angle(rQ)
        
        lsf_P = np.sort(anglesP[(anglesP > 1e-5) & (anglesP < np.pi - 1e-5)])
        lsf_Q = np.sort(anglesQ[(anglesQ > 1e-5) & (anglesQ < np.pi - 1e-5)])
        
        lsf = np.sort(np.concatenate([lsf_P, lsf_Q]))
        
        return lsf

    def lsf_to_lpc(self,lsf):
        lsf_sorted = np.sort(lsf)
        lsf_P = lsf_sorted[0::2]
        lsf_Q = lsf_sorted[1::2]
        
        def build_poly(freqs):
            poly = np.array([1.0])
            for w in freqs:
                quad = np.array([1.0, -2.0 * np.cos(w), 1.0])
                poly = np.convolve(poly, quad)
            return poly

        P_red  = build_poly(lsf_P)
        Q_red  = build_poly(lsf_Q)

        P_full = np.convolve(P_red, [1.0,  1.0])
        Q_full = np.convolve(Q_red, [1.0, -1.0])

        a = 0.5 * (P_full + Q_full)
        return a[:-1]
    
    def extract_mfcc(self, frames, fs, n_mfcc=13, n_filters=26):
        """Calculates MFCCs for a set of frames."""
        # 1. Power Spectrum
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of FFT
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))     # Power Spectrum

        # 2. Mel Filter Banks
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))      # Convert Mel back to Hz
        bin = np.floor((NFFT + 1) * hz_points / fs)

        fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, n_filters + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        # 3. DCT to get Coefficients
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        
        # 4. Mean Normalization (Optional but improves robustness)
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        return mfcc

class VectorialQuantization():
    def __init__(self):
        self.utils = VoiceUtils()
    def autocorr_   (self,a):
        p = len(a) - 1
        r_a = np.zeros(p + 1)
        for i in range(p + 1):
            r_a[i] = np.sum(a[:p + 1 - i] * a[i:])
        return r_a

    def itakura_saito_batch(self,autocorr_frames, centroids_lpc):
        p = centroids_lpc.shape[1] - 1
        r_a_all = np.array([self.autocorr_lpc(a) for a in centroids_lpc])
        weights = np.ones(p + 1)
        weights[1:] = 2.0
        # normalizar por energia del frame para quitar sesgo de amplitud
        r0 = np.maximum(autocorr_frames[:, 0:1], 1e-10)
        numerator = autocorr_frames[:, :p+1] @ (r_a_all * weights).T
        return numerator / r0

    def lbg(self,lsf_P_features, lsf_Q_features, autocorr_features,
            codebook_size, epsilon=1e-4, max_iter=100, delta=0.01):

        # centroides iniciales
        cent_P = np.array([lsf_P_features.mean(axis=0)])
        cent_Q = np.array([lsf_Q_features.mean(axis=0)])

        while len(cent_P) < codebook_size:
            # split
            cent_P = np.array([c * (1 + s * delta)
                            for c in cent_P for s in (+1, -1)])
            cent_Q = np.array([c * (1 + s * delta)
                            for c in cent_Q for s in (+1, -1)])

            prev_dist = None
            for _ in range(max_iter):
                # convertir a LPC para distancia IS
                centroids_lpc = np.array([
                    self.utils.lsf_to_lpc(np.sort(np.concatenate([cent_P[k], cent_Q[k]])))
                    for k in range(len(cent_P))
                ])

                dist_matrix = self.itakura_saito_batch(autocorr_features,
                                                centroids_lpc)
                assignments = np.argmin(dist_matrix, axis=1)
                total_dist  = dist_matrix[np.arange(len(dist_matrix)),
                                        assignments].sum()

                new_P, new_Q = [], []
                for k in range(len(cent_P)):
                    mask = assignments == k
                    if mask.sum() > 0:
                        new_P.append(lsf_P_features[mask].mean(axis=0))
                        new_Q.append(lsf_Q_features[mask].mean(axis=0))
                    else:
                        new_P.append(cent_P[k])
                        new_Q.append(cent_Q[k])

                cent_P = np.array(new_P)
                cent_Q = np.array(new_Q)

                if prev_dist is not None:
                    rel = abs(prev_dist - total_dist) / (abs(prev_dist) + 1e-10)
                    if rel < epsilon:
                        break
                prev_dist = total_dist

        return cent_P[:codebook_size], cent_Q[:codebook_size]


    def recognize(self,autocorr_frames, codebooks, word_labels):
        best_word, best_dist = None, np.inf

        for word in word_labels:
            cb_P, cb_Q = codebooks[word]
            cb_lpc = np.array([self.utils.lsf_to_lpc(np.sort(np.concatenate([cb_P[k], cb_Q[k]])))
                            for k in range(len(cb_P))])
            total = self.itakura_saito_batch(autocorr_frames,
                                        cb_lpc).min(axis=1).mean()
            if total < best_dist:
                best_dist, best_word = total, word

        return best_word