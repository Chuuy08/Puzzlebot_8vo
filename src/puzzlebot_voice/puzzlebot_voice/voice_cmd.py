#!/usr/bin/env python3
"""Main python node for the first step towards the voice recognition function"""
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from puzzlebot_voice import hmm_utils as _hmm_utils_mod
from puzzlebot_voice.voice_utils import VoiceUtils, VectorialQuantization
from scipy.signal import butter, filtfilt
from puzzlebot_voice.hmm_utils import HMMUtils as hmm
import random
import pickle
import threading

# hmm_best_model.pkl was pickled back when hmm_utils was a top-level module
# (HMMUtils -> "hmm_utils.HMMUtils"). Alias it so unpickling finds the class
# at its new location, without needing to retrain/re-save the model.
sys.modules.setdefault('hmm_utils', _hmm_utils_mod)

MODEL_PATH = os.path.join(
    get_package_share_directory('puzzlebot_voice'), 'models', 'hmm_best_model.pkl')

# - Lectura de la señal de audio (16KHz)
# - Filtro de preenfasis
# - Ventana de Hamming 
# - Inicio y final de cada palabra
# - 

# --- Template Matching
class VoiceCmdNode(Node):
    def __init__(self):
        super().__init__('void_cmd_node')
        self.media_path = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
        self.data, self.sr = self.load_dataset(self.media_path)
        self.codebook_sizes = [16, 32, 64]
        self.P = 12 
        self.alpha = 0.95
        self.codebooks = {}
        self.hmms = {}
        self.voice_pub = self.create_publisher(String, '/voice_cmd', 10)
        self.get_logger().info(f"Dataset loaded with {len(self.data)} words.")
        self.timer = self.create_timer(1.0, self.timer_callback)
           
    def timer_callback(self):
        utils = VoiceUtils()
        vectorial = VectorialQuantization()
        self.data, self.sr = self.load_dataset(self.media_path)
        self.train(utils, vectorial)

    def train(self, utils, vectorial):
        #self.load_dataset(os.path.join(get_package_share_directory('puzzlebot_voice'), 'media'))
        labels = list(self.data.keys())
        for size in self.codebook_sizes:
            self.get_logger().info(f"\nProbando con codebook de tamaño {size}")
            codebooks = {}
            for word in labels:
                features_all_P = []
                features_all_Q = []
                autocorr_all = []
                num_train_samples = min(35, len(self.data[word]))  # Use available samples, up to 10
                for i in range(num_train_samples):  # Training samples
                    fs, signal = self.data[word][i]
                    signal = utils.normalize(signal)
                    signal = utils.detect_voice(signal, fs)
                    if len(signal) < 320:
                        continue
                    signal = utils.pre_emphasis(signal,self.alpha)
                    frames = utils.framing(signal, fs)
                    if len(frames) == 0:
                        continue
                    frames = utils.hamming_window(frames)
                    lpc, _ = utils.extract_lpc(frames, self.P)
                    lsf = np.array([utils.lpc_to_lsf(l) for l in lpc])
                    lsf_P = lsf[:, :self.P//2]
                    lsf_Q = lsf[:, self.P//2:]
                    autocorrs = np.array([utils.autocorrelation(frame, self.P) for frame in frames])
                    features_all_P.extend(lsf_P)
                    features_all_Q.extend(lsf_Q)
                    autocorr_all.extend(autocorrs)
                if len(features_all_P) > 0:
                    codebooks[word] = vectorial.lbg(np.array(features_all_P), np.array(features_all_Q), np.array(autocorr_all), size)
            self.codebooks = codebooks
            self.test(utils, vectorial, lpc, lsf, lsf_P, lsf_Q)


    
    def test(self, utils, vectorial, lpc, lsf, lsf_P, lsf_Q):
        predictions = []
        true_labels = []
        for word in self.data.keys():
            num_train_samples = min(35, len(self.data[word]))
            num_test_start = num_train_samples
            num_test_samples = min(5, len(self.data[word]) - num_train_samples)
            for i in range(num_test_start, num_test_start + num_test_samples):  # Test samples
                fs, signal = self.data[word][i]
                signal = utils.normalize(signal)
                signal = utils.detect_voice(signal, fs)
                if len(signal) < 320:
                    continue
                signal = utils.pre_emphasis(signal, self.alpha)
                frames = utils.framing(signal, fs)
                if len(frames) == 0:
                    continue
                frames = utils.hamming_window(frames)
                autocorrs = np.array([utils.autocorrelation(frame, self.P) for frame in frames])
                pred_word = vectorial.recognize(autocorrs, self.codebooks, list(self.data.keys()))
                predictions.append(pred_word)
                true_labels.append(word)

        if len(predictions) == 0 or len(true_labels) == 0:
            self.get_logger().warning('No test samples available; skipping confusion matrix.')
            return

        labels = list(self.data.keys())
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        accuracy = np.trace(cm) / np.sum(cm)

        self.get_logger().info("Matriz de confusión: ")
        self.get_logger().info(f"{cm}")
        self.get_logger().info(f"Ṕrecisión: {accuracy:.2f}")
                                

    def load_dataset(self, path):
        data, sr = {}, {}
        for word in os.listdir(path):
            word_path = os.path.join(path, word)

            if not os.path.isdir(word_path):
                continue

            signals = []
            fs = []
            for file in sorted(os.listdir(word_path)):
                file_path = os.path.join(word_path, file)

                if not file.endswith(".wav"):
                    continue

                fs, signal = wav.read(file_path)
                if signal.ndim > 1:
                    signal = signal[:, 0]  
                signal = signal.astype(float) / 32768.0 # convert to float32 
                signals.append((fs, signal))

            data[word] = signals
            sr[word] = fs

        return data, sr

"""
class HMMTrainingNode_MFCC(Node):
    def __init__(self):
        super().__init__('void_cmd_node')
        self.media_path = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
        self.codebook_sizes = [32, 64, 128]
        self.alpha = 0.95
        self.global_centroids = None
        self.hmms = {}
        self._trained = False
        self._train_lock = threading.Lock()

        # Lazy load — solo paths, no señales en RAM
        self.data = self._index_dataset(self.media_path)

        random.seed(42)
        self.train_indices = {
            word: random.sample(range(len(paths)), min(35, len(paths)))
            for word, paths in self.data.items()
        }

        self.get_logger().info(f"Dataset indexado con {len(self.data)} palabras.")
        self.voice_pub = self.create_publisher(String, '/voice_cmd', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        if self._trained:
            return
        with self._train_lock:
            if self._trained:
                return
            self._trained = True

        # Si ya existe modelo en disco, cargarlo sin re-entrenar
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.hmms = data['hmms']
                self.global_centroids = data['global_centroids']
                self.get_logger().info(
                    f"Modelo cargado desde disco (size={data['codebook_size']}). Sin re-entrenamiento."
                )
                return
            except Exception as e:
                self.get_logger().warning(f"No se pudo cargar modelo: {e}. Re-entrenando.")

        # Entrenar en thread separado para no bloquear el executor de ROS2
        threading.Thread(target=self._train_background, daemon=True).start()

    def _train_background(self):
        self.get_logger().info("Iniciando entrenamiento en background thread...")
        utils = VoiceUtils()
        vectorial = VectorialQuantization()
        self.train(utils, vectorial)

    def _index_dataset(self, path):
        #Indexa solo los paths de los .wav, sin leer señales a RAM.
        data = {}
        for word in os.listdir(path):
            word_path = os.path.join(path, word)
            if not os.path.isdir(word_path):
                continue
            paths = [
                os.path.join(word_path, f)
                for f in sorted(os.listdir(word_path))
                if f.endswith('.wav')
            ]
            if paths:
                data[word] = paths
        return data

    def _read_sample(self, filepath):
        #Lee un .wav desde disco bajo demanda.
        fs, signal = wav.read(filepath)
        if signal.ndim > 1:
            signal = signal[:, 0]
        return fs, signal.astype(float) / 32768.0

    def preprocess_to_mfcc(self, signal, fs, utils):
        #Processes a raw signal into a matrix of MFCC vectors.
        signal = utils.normalize(signal)
        signal = utils.detect_voice(signal, fs)
        if len(signal) < 320:
            return None
        signal = utils.pre_emphasis(signal, self.alpha)
        frames = utils.framing(signal, fs)
        if len(frames) == 0:
            return None
        frames = utils.hamming_window(frames)
        return utils.extract_mfcc(frames, fs)

    def build_global_codebook(self, utils, size):
        #Generates a single universal VQ codebook using MFCCs from all words.
        all_mfccs = []
        for word, paths in self.data.items():
            for i in self.train_indices[word]:
                fs, signal = self._read_sample(paths[i])
                mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                if mfccs is not None:
                    all_mfccs.extend(mfccs)

        if not all_mfccs:
            return None

        # n_init=5 + k-means++ en lugar de n_init=35 — 7x más rápido, misma calidad
        kmeans = KMeans(n_clusters=size, n_init=5, init='k-means++', random_state=42)
        kmeans.fit(np.array(all_mfccs))
        return kmeans.cluster_centers_

    def quantize(self, mfcc_features, centroids):
        #Maps MFCC vectors to the closest global centroid index (Euclidean).
        diff = mfcc_features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return np.argmin(dists, axis=1)

    def train(self, utils, vectorial):
        labels = list(self.data.keys())
        best_accuracy = -1
        best_size = None
        best_hmms = {}
        best_centroids = None

        for size in self.codebook_sizes:
            self.get_logger().info(f"Training with codebook size: {size}")
            centroids = self.build_global_codebook(utils, size)
            if centroids is None:
                continue
            self.global_centroids = centroids

            hmms_candidate = {}
            for word in labels:
                sequences = []
                for i in self.train_indices[word]:
                    fs, signal = self._read_sample(self.data[word][i])
                    mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                    if mfccs is not None:
                        obs_seq = self.quantize(mfccs, centroids)
                        sequences.append(obs_seq)
                if sequences:
                    word_hmm = hmm(n_states=8, n_symbols=size)
                    word_hmm.train(sequences)
                    hmms_candidate[word] = word_hmm

            self.hmms = hmms_candidate
            accuracy = self.evaluate_hmms(utils, size)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_size = size
                best_hmms = hmms_candidate
                best_centroids = centroids
                self.get_logger().info(f"New best model: size={size}, accuracy={accuracy:.2f}")

        # Guardar pickle solo una vez al final, no en cada iteración
        if best_size is not None:
            self.hmms = best_hmms
            self.global_centroids = best_centroids
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'hmms': self.hmms,
                    'global_centroids': self.global_centroids,
                    'codebook_size': best_size
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.get_logger().info(f"Mejor tamaño: {best_size} con accuracy {best_accuracy:.2f}")
        else:
            self.get_logger().error("No se encontró ningún modelo válido.")
            

    def evaluate_hmms(self, utils, size):
        predictions, true_labels, confidences = [], [], []
        per_word_stats = {}

        for word, paths in self.data.items():
            word_scores, word_confs = [], []
            all_indices = list(range(len(paths)))
            test_indices = [i for i in all_indices if i not in self.train_indices[word]]

            for i in test_indices:
                fs, signal = self._read_sample(paths[i])
                mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                if mfccs is None:
                    continue

                obs_seq = self.quantize(mfccs, self.global_centroids)
                scores = {w: model.forward(obs_seq) for w, model in self.hmms.items()}
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                best_word, best_score = sorted_scores[0]
                second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1e100

                predictions.append(best_word)
                true_labels.append(word)
                word_scores.append(best_score)
                word_confs.append(best_score - second_score)

            if word_scores:
                per_word_stats[word] = {
                    'avg_score': np.mean(word_scores),
                    'avg_confidence': np.mean(word_confs),
                    'samples': len(word_scores)
                }

        if not predictions:
            self.get_logger().warning('No test samples available.')
            return 0.0

        cm = confusion_matrix(true_labels, predictions, labels=list(self.data.keys()))
        accuracy = np.trace(cm) / np.sum(cm)

        self.get_logger().info(f"[Size {size}] Accuracy: {accuracy:.2f}")
        self.get_logger().info(f"Confusion Matrix:\n{cm}")
        for word, stats in per_word_stats.items():
            self.get_logger().info(f"  '{word}': Score={stats['avg_score']:.2f}, Conf={stats['avg_confidence']:.2f}")

        return accuracy

"""
# --- HMM Node for training
class HMMTrainingNode_MFCC(Node):
    def __init__(self):
        super().__init__('void_cmd_node')
        self.media_path = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
        self.data, self.sr = self.load_dataset(self.media_path)
        self.codebook_sizes = [32, 64,128, 256]
        self.alpha = 0.95
        self.global_centroids = None
        self.hmms = {}
        self._trained = False

        # Fijar índices de train/test UNA sola vez
        random.seed(42)
        self.train_indices = {
            word: random.sample(range(len(samples)), min(35, len(samples)))
            for word, samples in self.data.items()
        }       

        self.get_logger().info(f"Dataset loaded with {len(self.data)} words.")
        self.voice_pub = self.create_publisher(String, '/voice_cmd', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        if self._trained:
            return
        utils = VoiceUtils()
        vectorial = VectorialQuantization()
        self.train(utils, vectorial)
        self._trained = True

    def preprocess_to_mfcc(self, signal, fs, utils):
        #Processes a raw signal into a matrix of MFCC vectors.
        signal = utils.normalize(signal)
        signal = utils.detect_voice(signal, fs)
        if len(signal) < 320:
            return None
        signal = utils.pre_emphasis(signal, self.alpha)
        frames = utils.framing(signal, fs)
        if len(frames) == 0:
            return None
        frames = utils.hamming_window(frames)
        return utils.extract_mfcc(frames, fs)

    def build_global_codebook(self, utils, size):
        #Generates a single universal VQ codebook using MFCCs from all words.
        all_mfccs = []
        for word in self.data.keys():
            for i in self.train_indices[word]:  # Usa los índices fijos
                fs, signal = self.data[word][i]
                mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                if mfccs is not None:
                    all_mfccs.extend(mfccs)

        if len(all_mfccs) == 0:
            return None

        kmeans = KMeans(n_clusters=size, n_init=35, random_state=42)
        kmeans.fit(np.array(all_mfccs))
        return kmeans.cluster_centers_

    def quantize(self, mfcc_features, centroids):
        #Maps MFCC vectors to the closest global centroid index (Euclidean).
        dist_matrix = cdist(mfcc_features, centroids, 'euclidean')
        return np.argmin(dist_matrix, axis=1)

    def train(self, utils, vectorial):
        labels = list(self.data.keys())
        best_accuracy = -1
        best_size = None

        for size in self.codebook_sizes:
            self.get_logger().info(f"Training with codebook size: {size}")
            self.global_centroids = self.build_global_codebook(utils, size)
            if self.global_centroids is None:
                continue

            hmms_candidate = {}
            for word in labels:
                sequences = []
                for i in self.train_indices[word]:  # Usa los índices fijos
                    fs, signal = self.data[word][i]
                    mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                    if mfccs is not None:
                        obs_seq = self.quantize(mfccs, self.global_centroids)
                        sequences.append(obs_seq)
                if sequences:
                    word_hmm = hmm(n_states=8, n_symbols=size)
                    word_hmm.train(sequences)
                    hmms_candidate[word] = word_hmm

            self.hmms = hmms_candidate
            accuracy = self.evaluate_hmms(utils, size)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_size = size
                with open('hmm_best_model.pkl', 'wb') as f:
                    pickle.dump({
                        'hmms': self.hmms,
                        'global_centroids': self.global_centroids,
                        'codebook_size': size
                    }, f)
                self.get_logger().info(f"New best model saved: size={size}, accuracy={accuracy:.2f}")

        self.get_logger().info(f"Mejor tamaño: {best_size} con accuracy {best_accuracy:.2f}")

    def evaluate_hmms(self, utils, size):
        predictions, true_labels, confidences = [], [], []
        per_word_stats = {}

        for word in self.data.keys():
            word_scores, word_confs = [], []
            all_indices = list(range(len(self.data[word])))
            test_indices = [i for i in all_indices if i not in self.train_indices[word]]

            for i in test_indices:
                fs, signal = self.data[word][i]
                mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                if mfccs is None:
                    continue

                obs_seq = self.quantize(mfccs, self.global_centroids)
                scores = {w: model.forward(obs_seq) for w, model in self.hmms.items()}
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                best_word, best_score = sorted_scores[0]
                second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1e100

                predictions.append(best_word)
                true_labels.append(word)
                word_scores.append(best_score)
                word_confs.append(best_score - second_score)

            if word_scores:
                per_word_stats[word] = {
                    'avg_score': np.mean(word_scores),
                    'avg_confidence': np.mean(word_confs),
                    'samples': len(word_scores)
                }

        if not predictions:
            self.get_logger().warning('No test samples available.')
            return 0.0

        cm = confusion_matrix(true_labels, predictions, labels=list(self.data.keys()))
        accuracy = np.trace(cm) / np.sum(cm)

        self.get_logger().info(f"[Size {size}] Accuracy: {accuracy:.2f}")
        self.get_logger().info(f"Confusion Matrix:\n{cm}")
        for word, stats in per_word_stats.items():
            self.get_logger().info(f"  '{word}': Score={stats['avg_score']:.2f}, Conf={stats['avg_confidence']:.2f}")

        return accuracy

    def load_dataset(self, path):
        data, sr = {}, {}
        for word in os.listdir(path):
            word_path = os.path.join(path, word)
            if not os.path.isdir(word_path):
                continue
            signals = []
            fs = []
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
            sr[word] = fs
        return data, sr
 
 
   
class HMMTrainingNode_LPC(Node):
    def __init__(self):
        super().__init__('void_cmd_node')
        self.media_path = os.path.join(get_package_share_directory('puzzlebot_voice'), 'media', 'audio')
        self.data, self.sr = self.load_dataset(self.media_path)
        self.codebook_sizes = [16, 32, 64,128, 256]
        self.P = 12 
        self.alpha = 0.95
        self.codebooks = {}
        self.get_logger().info(f"Dataset loaded with {len(self.data)} words.")
        self.timer = self.create_timer(1.0, self.timer_callback)
           
    def timer_callback(self):
        utils = VoiceUtils()
        vectorial = VectorialQuantization()
        self.data, self.sr = self.load_dataset(self.media_path)
        self.train(utils, vectorial)

    def preprocess(self, signal, fs, utils):
        signal = utils.normalize(signal)
        signal = utils.detect_voice(signal, fs)
        if len(signal) < 320:
            return None
        signal = utils.pre_emphasis(signal, self.alpha)
        frames = utils.framing(signal, fs)
        if len(frames) == 0:
            return None
        frames = utils.hamming_window(frames)
        autocorrs = np.array([utils.autocorrelation(frame, self.P) for frame in frames])
        return {'autocorrs': autocorrs}
    
    def build_global_codebook(self, utils, vectorial, size):
        all_P = []
        all_Q = []
        all_autocorr = []
        all_mfcc = []
        for word in self.data.keys():
            num_train_samples = min(35, len(self.data[word]))
            for i in range(num_train_samples):
                fs, signal = self.data[word][i]
                processed = self.preprocess(signal, fs, utils)
                if processed is None:
                    continue
                frames = utils.framing(utils.pre_emphasis(utils.detect_voice(signal, fs), self.alpha), fs)
                if len(frames) == 0:
                    continue
                lpc, _ = utils.extract_lpc(frames, self.P)
                lsf = np.array([utils.lpc_to_lsf(l) for l in lpc])
                all_P.extend(lsf[:, :self.P//2])
                all_Q.extend(lsf[:, self.P//2:])
                all_autocorr.extend(processed['autocorrs'])

        if len(all_P) == 0:
            return None, None, None

        cent_P, cent_Q = vectorial.lbg(
            np.array(all_P),
            np.array(all_Q),
            np.array(all_autocorr),
            size)
        centroids_lpc = np.array([
            utils.lsf_to_lpc(np.concatenate([cent_P[k], cent_Q[k]]))
            for k in range(len(cent_P))])
        return cent_P, cent_Q, centroids_lpc #ecuclidian distance MFFCCs

    def quantize(self, autocorrs, centroids_lpc, vectorial):
        dist_matrix = vectorial.itakura_saito_batch(autocorrs, centroids_lpc)
        return np.argmin(dist_matrix, axis=1)

    def evaluate_hmms(self, utils, vectorial, size):
        predictions = []
        true_labels = []
        confidences = []
        per_word_stats = {}

        for word in self.data.keys():
            word_scores = []
            word_confs = []
            for fs, signal in self.data[word][35:]:
                processed = self.preprocess(signal, fs, utils)
                if processed is None:
                    continue
                obs_seq = self.quantize(processed['autocorrs'], self.global_lpc_centroids, vectorial)
                if len(obs_seq) == 0:
                    continue

                scores = {w: model.forward(obs_seq) for w, model in self.hmms.items()}
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_word, best_score = sorted_scores[0]
                second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
                confidence = best_score - second_score

                predictions.append(best_word)
                true_labels.append(word)
                confidences.append(confidence)
                word_scores.append(best_score)
                word_confs.append(confidence)

            if word_scores:
                per_word_stats[word] = {
                    'avg_score': float(np.mean(word_scores)),
                    'avg_confidence': float(np.mean(word_confs)),
                    'samples': len(word_scores)
                }

        if len(predictions) == 0 or len(true_labels) == 0:
            self.get_logger().warning('No evaluation data available for confidence testing.')
            return

        cm = confusion_matrix(true_labels, predictions, labels=list(self.data.keys()))
        accuracy = np.trace(cm) / np.sum(cm)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        self.get_logger().info(f"Evaluation for codebook size {size}")
        self.get_logger().info(f"Overall accuracy: {accuracy:.2f}")
        self.get_logger().info(f"Average confidence gap: {avg_confidence:.4f}")
        self.get_logger().info(f"Confusion matrix:\n{cm}")

        for word, stats in per_word_stats.items():
            self.get_logger().info(
                f"Word '{word}': avg score={stats['avg_score']:.4f}, "
                f"avg confidence={stats['avg_confidence']:.4f}, "
                f"samples={stats['samples']}")

    def train(self, utils, vectorial):
        labels = list(self.data.keys())
        best_accuracy = -1
        best_size = None

        for size in self.codebook_sizes:
            self.global_centroids = self.build_global_codebook(utils, size)
            if self.global_centroids is None:
                continue

            hmms_candidate = {}
            for word in labels:
                sequences = []
                train_indices = random.sample(range(len(self.data[word])), min(35, len(self.data[word])))
                self.train_indices[word] = train_indices
                for i in train_indices:
                    fs, signal = self.data[word][i]
                    mfccs = self.preprocess_to_mfcc(signal, fs, utils)
                    if mfccs is not None:
                        obs_seq = self.quantize(mfccs, self.global_centroids)
                        sequences.append(obs_seq)
                if sequences:
                    word_hmm = hmm(n_states=8, n_symbols=size)
                    word_hmm.train(sequences)
                    hmms_candidate[word] = word_hmm

            self.hmms = hmms_candidate
            accuracy = self.evaluate_hmms(utils, size)  # <-- que retorne el valor

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_size = size
                # Guardar el mejor modelo
                with open('hmm_best_model.pkl', 'wb') as f:
                    pickle.dump({
                        'hmms': self.hmms,
                        'global_centroids': self.global_centroids,
                        'codebook_size': size
                    }, f)

        self.get_logger().info(f"Mejor tamaño: {best_size} con accuracy {best_accuracy:.2f}")


    def test(self, utils, vectorial, lpc, lsf, lsf_P, lsf_Q):
        predictions = []
        true_labels = []
        for word in self.data.keys():
            num_train_samples = min(35, len(self.data[word]))
            num_test_start = num_train_samples
            num_test_samples = min(5, len(self.data[word]) - num_train_samples)
            for i in range(num_test_start, num_test_start + num_test_samples):  # Test samples
                fs, signal = self.data[word][i]
                signal = utils.normalize(signal)
                signal = utils.detect_voice(signal, fs)
                if len(signal) < 320:
                    continue
                signal = utils.pre_emphasis(signal, self.alpha)
                frames = utils.framing(signal, fs)
                if len(frames) == 0:
                    continue
                frames = utils.hamming_window(frames)
                autocorrs = np.array([utils.autocorrelation(frame, self.P) for frame in frames])
                pred_word = vectorial.recognize(autocorrs, self.codebooks, list(self.data.keys()))
                predictions.append(pred_word)
                true_labels.append(word)

        if len(predictions) == 0 or len(true_labels) == 0:
            self.get_logger().warning('No test samples available; skipping confusion matrix.')
            return

        labels = list(self.data.keys())
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        accuracy = np.trace(cm) / np.sum(cm)

        self.get_logger().info("Matriz de confusión: ")
        self.get_logger().info(f"{cm}")
        self.get_logger().info(f"Ṕrecisión: {accuracy:.2f}")
    
    def recognize(self, test_signal_features):
        best_word = None
        max_log_prob = -np.inf
        
        # Convert test audio frames to a sequence of indices (O)
        obs_seq = self.vectorial_quantizer(test_signal_features)
        
        for word, model in self.hmms.items():
            # Solve Problem 1: Score the model
            prob = model.forward(obs_seq)
            
            if prob > max_log_prob:
                max_log_prob = prob
                best_word = word
                
        return best_word
                                

    def load_dataset(self, path):
        data, sr = {}, {}
        for word in os.listdir(path):
            word_path = os.path.join(path, word)

            if not os.path.isdir(word_path):
                continue

            signals = []
            fs = []
            for file in sorted(os.listdir(word_path)):
                file_path = os.path.join(word_path, file)

                if not file.endswith(".wav"):
                    continue

                fs, signal = wav.read(file_path)
                if signal.ndim > 1:
                    signal = signal[:, 0]  
                signal = signal.astype(float) / 32768.0 # convert to float32 
                signals.append((fs, signal))

            data[word] = signals
            sr[word] = fs

        return data, sr       
    

def main(args=None):
    rclpy.init(args=args)
    #voice_cmd_node = VoiceCmdNode()
    voice_cmd_node = HMMTrainingNode_MFCC()
    rclpy.spin_once(voice_cmd_node) 
    voice_cmd_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()