import numpy as np
from .features import lsf_to_lpc

def autocorr_lpc(a):
    p = len(a) - 1
    r_a = np.zeros(p + 1)
    for i in range(p + 1):
        r_a[i] = np.sum(a[:p + 1 - i] * a[i:])
    return r_a

def itakura_saito_batch(autocorr_frames, centroids_lpc):
    p = centroids_lpc.shape[1] - 1
    r_a_all = np.array([autocorr_lpc(a) for a in centroids_lpc])
    weights = np.ones(p + 1)
    weights[1:] = 2.0
    # normalizar por energia del frame para quitar sesgo de amplitud
    r0 = np.maximum(autocorr_frames[:, 0:1], 1e-10)
    numerator = autocorr_frames[:, :p+1] @ (r_a_all * weights).T
    return numerator / r0

def lbg(lsf_P_features, lsf_Q_features, autocorr_features,
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
                lsf_to_lpc(np.sort(np.concatenate([cent_P[k], cent_Q[k]])))
                for k in range(len(cent_P))
            ])

            dist_matrix = itakura_saito_batch(autocorr_features,
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


def recognize(autocorr_frames, codebooks, word_labels):
    best_word, best_dist = None, np.inf

    for word in word_labels:
        cb_P, cb_Q = codebooks[word]
        cb_lpc = np.array([lsf_to_lpc(np.sort(np.concatenate([cb_P[k], cb_Q[k]])))
                           for k in range(len(cb_P))])
        total = itakura_saito_batch(autocorr_frames,
                                    cb_lpc).min(axis=1).mean()
        if total < best_dist:
            best_dist, best_word = total, word

    return best_word