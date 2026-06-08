import numpy as np

def autocorrelation(frame, p):
    r = np.zeros(p + 1)
    for k in range(p + 1):
        r[k] = np.sum(frame[:len(frame) - k] * frame[k:])
    return r

def levinson_durbin(r, p):
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

def extract_lpc(frames, order=12):
    lpc_vectors, sigmas = [], []
    for frame in frames:
        r = autocorrelation(frame, order)
        a, s = levinson_durbin(r, order)
        lpc_vectors.append(a)
        sigmas.append(s)
    return np.array(lpc_vectors), np.array(sigmas)

def lpc_to_lsf(a):
    """
    Convierte LPC a LSF usando raíces de polinomios P y Q.
    """
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

def lsf_to_lpc(lsf):
    """
    Convierte LSF a LPC.
    """
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