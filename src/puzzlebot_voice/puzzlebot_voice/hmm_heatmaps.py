"""
hmm_heatmaps.py
Genera heatmaps de matrices A y B en 3 momentos del entrenamiento
para 3 palabras seleccionadas del vocabulario HMM.

Uso:
    python3 hmm_heatmaps.py

Requiere: numpy, matplotlib, seaborn, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.io import wavfile as wav
import os
import random
import copy

from ament_index_python.packages import get_package_share_directory

# ── Reproducir exactamente el mismo split que el nodo ROS ─────────────────────
random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
#  CLASES HMM  (copia exacta de tu implementación)
# ═════════════════════════════════════════════════════════════════════════════

class HMMUtils:
    def __init__(self, n_states, n_symbols):
        self.N = n_states
        self.M = n_symbols

        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i < n_states - 1:
                self.A[i, i]   = 0.5
                self.A[i, i+1] = 0.5
            else:
                self.A[i, i] = 1.0

        self.B = np.full((n_states, n_symbols), 1.0 / n_symbols)

        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0

    def viterbi(self, obs_seq):
        T = len(obs_seq)
        if T == 0:
            return np.array([], dtype=int)

        log_A  = np.log(self.A  + 1e-300)
        log_B  = np.log(self.B  + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, self.N))
        phi   = np.zeros((T, self.N), dtype=int)

        delta[0, :] = log_pi + log_B[:, obs_seq[0]]

        for t in range(1, T):
            trans     = delta[t-1, :, None] + log_A
            phi[t, :] = np.argmax(trans, axis=0)
            delta[t, :] = np.max(trans, axis=0) + log_B[:, obs_seq[t]]

        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            states[t] = phi[t+1, states[t+1]]
        return states

    def train_with_snapshots(self, sequences, max_iter=20):
        """
        Igual que train() pero devuelve snapshots de B en 3 momentos:
          - 'initial'      : B justo después de contar (iter 0, antes de normalizar)
          - 'intermediate' : B a la mitad del entrenamiento
          - 'final'        : B al terminar (con suavizado Laplace ya aplicado)

        Returns
        -------
        snapshots : dict con keys 'initial', 'intermediate', 'final'
                    cada valor es un np.ndarray (N, M)
        """
        mid_iter    = max_iter // 2          # iteración "intermedia"
        snapshots   = {}
        converged   = False

        for iteration in range(max_iter):
            new_B = np.ones((self.N, self.M)) * 1e-6   # Laplace smoothing

            for seq in sequences:
                states = self.viterbi(seq)
                for t, s in enumerate(states):
                    new_B[s, seq[t]] += 1

            # ── Snapshot INICIAL: conteos crudos antes de normalizar (iter 0) ──
            if iteration == 0:
                snapshots['initial'] = new_B.copy()   # sin normalizar aún

            new_B /= new_B.sum(axis=1, keepdims=True)

            # ── Snapshot INTERMEDIO ──────────────────────────────────────────
            if iteration == mid_iter:
                snapshots['intermediate'] = new_B.copy()

            if np.max(np.abs(new_B - self.B)) < 1e-4:
                print(f"  Converged at iteration {iteration}")
                converged = True
                self.B = new_B
                break

            self.B = new_B

        # ── Snapshot FINAL ───────────────────────────────────────────────────
        snapshots['final'] = self.B.copy()

        # Si convergió antes del punto medio, rellena 'intermediate' con final
        if 'intermediate' not in snapshots:
            snapshots['intermediate'] = snapshots['final'].copy()

        return snapshots


# ═════════════════════════════════════════════════════════════════════════════
#  PROCESAMIENTO DE SEÑAL  (copia exacta de VoiceUtils simplificada)
# ═════════════════════════════════════════════════════════════════════════════

class VoiceUtils:
    def normalize(self, signal):
        mx = np.max(np.abs(signal))
        return signal / mx if mx > 0 else signal

    def detect_voice(self, signal, fs, frame_ms=20, threshold=0.02):
        frame_len = int(fs * frame_ms / 1000)
        voiced = []
        for start in range(0, len(signal) - frame_len, frame_len):
            frame = signal[start:start + frame_len]
            if np.sqrt(np.mean(frame**2)) > threshold:
                voiced.extend(frame)
        return np.array(voiced) if voiced else signal

    def pre_emphasis(self, signal, alpha=0.95):
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])

    def framing(self, signal, fs, frame_ms=25, step_ms=10):
        frame_len = int(fs * frame_ms / 1000)
        step      = int(fs * step_ms  / 1000)
        n_frames  = 1 + (len(signal) - frame_len) // step
        frames = np.array([
            signal[i*step : i*step + frame_len]
            for i in range(n_frames)
            if i*step + frame_len <= len(signal)
        ])
        return frames

    def hamming_window(self, frames):
        return frames * np.hamming(frames.shape[1])

    def extract_mfcc(self, frames, fs, n_mfcc=13, n_filters=26, n_fft=512):
        from numpy.fft import rfft
        pre_emphasis_coeff = 0.97
        emphasis = np.append(frames[:, 0:1],
                             frames[:, 1:] - pre_emphasis_coeff * frames[:, :-1],
                             axis=1)

        mag  = np.abs(rfft(emphasis, n=n_fft))
        power = (1.0 / n_fft) * (mag ** 2)

        low_mel  = 0
        high_mel = 2595 * np.log10(1 + (fs / 2) / 700)
        mel_pts  = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
        bin_pts  = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
        bin_pts  = np.clip(bin_pts, 0, power.shape[1] - 1)

        fbank = np.zeros((n_filters, power.shape[1]))
        for m in range(1, n_filters + 1):
            f_m_minus = bin_pts[m-1]
            f_m       = bin_pts[m]
            f_m_plus  = bin_pts[m+1]
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        filter_banks = np.dot(power, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        mfcc = np.zeros((frames.shape[0], n_mfcc))
        for n in range(n_mfcc):
            mfcc[:, n] = np.sum(
                filter_banks * np.cos(np.pi * n / n_filters * (np.arange(n_filters) + 0.5)),
                axis=1
            )

        # Delta MFCCs
        delta = np.zeros_like(mfcc)
        for t in range(1, mfcc.shape[0] - 1):
            delta[t] = (mfcc[t+1] - mfcc[t-1]) / 2

        return np.concatenate([mfcc, delta], axis=1)


class VectorialQuantization:
    pass


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

MEDIA_PATH    = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
SELECTED_WORDS = ['forward', 'right', 'start']                    # None = auto-selecciona las primeras 3
N_STATES      = 8
CODEBOOK_SIZE = 128                      # usa el mejor codebook de tu modelo
ALPHA         = 0.95


def load_dataset(path):
    data = {}
    for word in sorted(os.listdir(path)):
        word_path = os.path.join(path, word)
        if not os.path.isdir(word_path):
            continue
        signals = []
        for file in sorted(os.listdir(word_path)):
            if not file.endswith(".wav"):
                continue
            fs, signal = wav.read(os.path.join(word_path, file))
            if signal.ndim > 1:
                signal = signal[:, 0]
            signal = signal.astype(float) / 32768.0
            signals.append((fs, signal))
        if signals:
            data[word] = signals
    return data


def preprocess(signal, fs, utils, alpha=0.95):
    signal = utils.normalize(signal)
    signal = utils.detect_voice(signal, fs)
    if len(signal) < 320:
        return None
    signal = utils.pre_emphasis(signal, alpha)
    frames = utils.framing(signal, fs)
    if len(frames) == 0:
        return None
    frames = utils.hamming_window(frames)
    return utils.extract_mfcc(frames, fs)


def build_codebook(data, train_indices, utils, size):
    from sklearn.cluster import KMeans
    all_mfccs = []
    for word, samples in data.items():
        for i in train_indices[word]:
            fs, signal = samples[i]
            mfccs = preprocess(signal, fs, utils)
            if mfccs is not None:
                all_mfccs.extend(mfccs)
    km = KMeans(n_clusters=size, n_init=35, random_state=42)
    km.fit(np.array(all_mfccs))
    return km.cluster_centers_


def quantize(mfccs, centroids):
    from scipy.spatial.distance import cdist
    dist = cdist(mfccs, centroids, 'euclidean')
    return np.argmin(dist, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

MOMENTS = ['initial', 'intermediate', 'final']
MOMENT_LABELS = {
    'initial':      'Inicial\n(conteos post-segmentación)',
    'intermediate': 'Intermedia\n(mitad del refinamiento)',
    'final':        'Final\n(con suavizado aplicado)',
}


def plot_heatmaps(word, A_matrix, B_snapshots, out_dir="."):
    """
    Genera una figura con 6 heatmaps: A y B × 3 momentos.
    """
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('#0d0d0d')

    # Título general
    fig.suptitle(
        f'Evolución de Matrices HMM  —  Palabra: "{word.upper()}"',
        fontsize=18, fontweight='bold', color='#e8e0d0',
        y=0.98
    )

    # Grid: 2 filas (A, B) × 3 columnas (momentos)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97,
                           top=0.90, bottom=0.06)

    cmap_A = 'YlOrRd'
    cmap_B = 'Blues'

    for col, moment in enumerate(MOMENTS):
        # ── Matriz A (fija, left-right) ─────────────────────────────────────
        ax_A = fig.add_subplot(gs[0, col])
        sns.heatmap(
            A_matrix,
            ax=ax_A,
            cmap=cmap_A,
            linewidths=0.4,
            linecolor='#1a1a1a',
            annot=(N_STATES <= 10),
            fmt='.2f',
            annot_kws={'size': 7, 'color': '#1a1a1a'},
            cbar_kws={'shrink': 0.8},
            vmin=0, vmax=1,
        )
        ax_A.set_title(MOMENT_LABELS[moment], fontsize=9,
                       color='#c0b090', pad=6)
        ax_A.set_xlabel('Estado destino j', fontsize=8, color='#888')
        ax_A.set_ylabel('Estado origen i', fontsize=8, color='#888')
        ax_A.tick_params(colors='#888', labelsize=7)
        ax_A.set_facecolor('#111')
        if col == 0:
            ax_A.set_ylabel('Matriz A\n\nEstado origen i',
                            fontsize=9, color='#e8a040', fontweight='bold')

        # ── Matriz B ────────────────────────────────────────────────────────
        ax_B = fig.add_subplot(gs[1, col])
        B = B_snapshots[moment]
        # Para B con 256 columnas no ponemos anotaciones numéricas
        sns.heatmap(
            B,
            ax=ax_B,
            cmap=cmap_B,
            linewidths=0,
            cbar_kws={'shrink': 0.8},
            vmin=0, vmax=B.max(),
        )
        ax_B.set_xlabel(f'Símbolo VQ  (M={B.shape[1]})', fontsize=8, color='#888')
        ax_B.set_ylabel('Estado i', fontsize=8, color='#888')
        ax_B.tick_params(colors='#888', labelsize=7)
        ax_B.set_facecolor('#111')
        if col == 0:
            ax_B.set_ylabel('Matriz B\n\nEstado i',
                            fontsize=9, color='#4090e0', fontweight='bold')

        # Colorbar estilo
        for ax in [ax_A, ax_B]:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(colors='#888', labelsize=7)
                cbar.ax.yaxis.label.set_color('#888')

    # Análisis de una línea (sparsity de B final)
    B_final = B_snapshots['final']
    nonzero_ratio = np.count_nonzero(B_final > 1e-4) / B_final.size
    if nonzero_ratio < 0.3:
        analysis = (f'La matriz B final muestra picos fonéticos claros '
                    f'({nonzero_ratio*100:.1f}% de celdas activas) → especialización conservada.')
    else:
        analysis = (f'La matriz B final se dispersó ampliamente '
                    f'({nonzero_ratio*100:.1f}% de celdas activas) → distribución difusa.')

    fig.text(0.5, 0.01, f'📊 {analysis}',
             ha='center', va='bottom', fontsize=9,
             color='#a0d080', style='italic',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#1a2a1a', edgecolor='#3a5a3a', alpha=0.8))

    out_path = os.path.join(out_dir, f'hmm_heatmap_{word}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Guardado: {out_path}")
    return analysis


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    utils = VoiceUtils()

    print("Cargando dataset...")
    data = load_dataset(MEDIA_PATH)
    if not data:
        raise FileNotFoundError(f"No se encontraron datos en: {MEDIA_PATH}")

    all_words = list(data.keys())
    print(f"Palabras disponibles: {all_words}")

    # Selección de 3 palabras
    words_to_plot = SELECTED_WORDS or all_words[:3]
    print(f"Palabras seleccionadas: {words_to_plot}")

    # Train/test split idéntico al nodo ROS
    train_indices = {
        word: random.sample(range(len(samples)), min(35, len(samples)))
        for word, samples in data.items()
    }

    print(f"\nConstruyendo codebook global (size={CODEBOOK_SIZE})...")
    centroids = build_codebook(data, train_indices, utils, CODEBOOK_SIZE)

    os.makedirs("heatmaps_output", exist_ok=True)
    analyses = {}

    for word in words_to_plot:
        print(f"\n── Entrenando HMM para: '{word}' ──")
        sequences = []
        for i in train_indices[word]:
            fs, signal = data[word][i]
            mfccs = preprocess(signal, fs, utils, ALPHA)
            if mfccs is not None:
                obs_seq = quantize(mfccs, centroids)
                sequences.append(obs_seq)

        print(f"  Secuencias de entrenamiento: {len(sequences)}")
        model = HMMUtils(N_STATES, CODEBOOK_SIZE)
        B_snapshots = model.train_with_snapshots(sequences, max_iter=20)

        analysis = plot_heatmaps(
            word,
            model.A,
            B_snapshots,
            out_dir="heatmaps_output"
        )
        analyses[word] = analysis

    # Resumen final
    print("\n" + "="*60)
    print("ANÁLISIS DE UNA LÍNEA POR PALABRA")
    print("="*60)
    for word, text in analyses.items():
        print(f"  {word:15s}: {text}")


if __name__ == '__main__':
    main()