#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import sounddevice as sd
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from puzzlebot_voice import hmm_utils
from puzzlebot_voice.voice_utils import VoiceUtils
from scipy.signal import butter, filtfilt

# hmm_best_model.pkl was pickled back when hmm_utils was a top-level module
# (HMMUtils -> "hmm_utils.HMMUtils"). Alias it so unpickling finds the class
# at its new location, without needing to retrain/re-save the model.
sys.modules.setdefault('hmm_utils', hmm_utils)

class VoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('voice_recognition_node')
        model_path = os.path.join(
            get_package_share_directory('puzzlebot_voice'), 'models', 'hmm_best_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.hmms = model_data['hmms']
        self.global_centroids = np.ascontiguousarray(
            model_data['global_centroids'], dtype=np.float32)
        self.alpha = 0.95
        self.fs_record = 16000
        self.fs_model  = 16000
        self.duration  = 1.0
        self.device    = 9
        self.min_confidence = 5.0
        self.utils = VoiceUtils()
        self.voice_pub = self.create_publisher(String, '/voice_cmd', 10)
        self.get_logger().info("Model loaded.")

    # ------------------------------------------------------------------ #
    #  Audio capture                                                       #
    # ------------------------------------------------------------------ #
    def record(self):
        self.get_logger().info("Recording...")
        audio = sd.rec(
            int(self.duration * self.fs_record),
            samplerate=self.fs_record,
            channels=2,
            device=self.device,
            dtype='float32'
        )
        sd.wait()
        audio = audio.mean(axis=1)
        #b, a = butter(4, 80, btype='high', fs=self.fs_record)
        #audio = filtfilt(b, a, audio)
        return audio

    # ------------------------------------------------------------------ #
    #  VQ quantization                                                     #
    # ------------------------------------------------------------------ #
    def _quantize(self, mfccs):
        mfccs = mfccs.astype(np.float32)
        mfcc_sq = np.sum(mfccs ** 2, axis=1, keepdims=True)
        cent_sq = np.sum(self.global_centroids ** 2, axis=1)
        cross   = mfccs @ self.global_centroids.T
        dists   = mfcc_sq + cent_sq - 2 * cross
        return np.argmin(dists, axis=1)

    # ------------------------------------------------------------------ #
    #  Recognition (returns word + raw gap, no threshold applied)         #
    # ------------------------------------------------------------------ #
    def recognize(self, signal, apply_threshold=True):
        signal = self.utils.normalize(signal)
        signal = self.utils.detect_voice(signal, self.fs_model)

        self.get_logger().info(
            f"VAD: {len(signal)} samples ({len(signal)/self.fs_model:.3f}s)")

        if len(signal) < 320:
            self.get_logger().warning("Signal too short after VAD")
            return None, None

        signal = self.utils.pre_emphasis(signal, self.alpha)
        frames = self.utils.framing(signal, self.fs_model)
        if len(frames) == 0:
            return None, None

        frames   = self.utils.hamming_window(frames)
        mfcc     = self.utils.extract_mfcc(frames, self.fs_model)
        obs_seq  = self._quantize(mfcc.astype(np.float32))

        scores       = {w: model.forward(obs_seq) for w, model in self.hmms.items()}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for w, s in sorted_scores:
            self.get_logger().info(f"  {w}: {s:.2f}")

        best_word  = sorted_scores[0][0]
        confidence = sorted_scores[0][1] - sorted_scores[1][1]
        self.get_logger().info(f">>> {best_word} (gap: {confidence:.2f})")

        if apply_threshold and confidence < self.min_confidence:
            self.get_logger().warning(
                f"Low confidence ({confidence:.2f}), ignoring")
            return None, confidence

        return best_word, confidence

    # ------------------------------------------------------------------ #
    #  Normal inference loop                                               #
    # ------------------------------------------------------------------ #
    def run(self):
        while rclpy.ok():
            input("\nPress ENTER to record...")
            signal = self.record()
            word, _ = self.recognize(signal)
            if word is not None:
                msg = String()
                msg.data = word
                self.voice_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Evaluation mode: records N samples per word, prints confusion matrix#
    # ------------------------------------------------------------------ #
    def run_evaluation(self, n_per_word=10):
        """
        Interactive evaluation loop.
        Asks the user to say each word N times, then prints a confusion
        matrix and per-word accuracy — same logic as online inference,
        same microphone, same pipeline.
        """
        labels     = sorted(self.hmms.keys())
        true_labels  = []
        pred_labels  = []
        confidences  = []
        rejected     = 0

        self.get_logger().info(
            f"\n{'='*55}\n  EVALUATION MODE — {n_per_word} samples per word\n{'='*55}")
        self.get_logger().info(f"Words: {labels}\n")

        for word in labels:
            self.get_logger().info(
                f"\n--- Word: '{word.upper()}' ({n_per_word} recordings) ---")
            collected = 0
            while collected < n_per_word:
                input(f"  [{collected+1}/{n_per_word}] Press ENTER and say '{word}'...")
                signal = self.record()

                # apply_threshold=False: we log everything, reject nothing
                pred, gap = self.recognize(signal, apply_threshold=False)

                if pred is None:
                    self.get_logger().warning("  → Signal too short, not counted. Try again.")
                    continue

                true_labels.append(word)
                pred_labels.append(pred)
                confidences.append(gap)

                status = "✓" if pred == word else f"✗ (predicted: {pred})"
                self.get_logger().info(f"  → {status}  gap={gap:.2f}")
                collected += 1

        # ---- Build and print confusion matrix ---- #
        self._print_confusion_matrix(true_labels, pred_labels,
                                     confidences, labels, rejected)

    def _print_confusion_matrix(self, true_labels, pred_labels,
                                 confidences, labels, rejected):
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        label_idx = {l: i for i, l in enumerate(labels)}

        for t, p in zip(true_labels, pred_labels):
            cm[label_idx[t]][label_idx[p]] += 1

        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        avg_conf = np.mean(confidences) if confidences else 0.0

        sep  = "=" * 55
        col_w = max(len(l) for l in labels) + 2   # column width

        lines = [
            "",
            sep,
            "  CONFUSION MATRIX (rows=true, cols=predicted)",
            sep,
            "          " + "".join(f"{l:>{col_w}}" for l in labels),
            "         " + "-" * (col_w * n + 1),
        ]

        for i, true_w in enumerate(labels):
            row_str = f"  {true_w:>8} |"
            for j in range(n):
                val = cm[i][j]
                # mark diagonal
                cell = f"[{val}]" if i == j else f" {val} "
                row_str += f"{cell:>{col_w}}"
            lines.append(row_str)

        lines += [
            sep,
            f"  Overall accuracy : {accuracy*100:.1f}%",
            f"  Avg gap (conf.)  : {avg_conf:.2f}",
            f"  Samples total    : {np.sum(cm)}",
            sep,
        ]

        # Per-word accuracy
        lines.append("  Per-word accuracy:")
        for i, word in enumerate(labels):
            total = cm[i].sum()
            correct = cm[i][i]
            pct = 100 * correct / total if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"    {word:>10}: {correct:2}/{total}  {pct:5.1f}%  {bar}")

        # Confidence distribution hint
        lines.append(f"\n  Gap threshold currently: {self.min_confidence}")
        gaps = np.array(confidences)
        lines.append(f"  Gap min/median/max: "
                     f"{gaps.min():.1f} / {np.median(gaps):.1f} / {gaps.max():.1f}")
        would_reject = np.sum(gaps < self.min_confidence)
        lines.append(f"  Samples below threshold: {would_reject}/{len(gaps)} "
                     f"({100*would_reject/len(gaps):.0f}%) would be rejected in normal mode")
        lines.append(sep)

        report = "\n".join(lines)
        self.get_logger().info(report)
        print(report)   # also print to stdout for easy copy-paste


def main(args=None):
    import sys
    rclpy.init(args=args)
    node = VoiceRecognitionNode()

    # Pass --eval [N] to enter evaluation mode
    # Example:  ros2 run puzzlebot_control voice_node --eval 10
    eval_mode = '--eval' in sys.argv
    if eval_mode:
        try:
            idx = sys.argv.index('--eval')
            n = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            n = 5
        node.run_evaluation(n_per_word=n)
    else:
        node.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()