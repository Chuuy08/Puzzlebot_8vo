import sqlite3
import struct
import numpy as np
import glob
import os

bag_path = './calibracion_bag'
db_file = glob.glob(os.path.join(bag_path, '*.db3'))[0]

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

cursor.execute("SELECT id, name FROM topics")
topics = {name: tid for tid, name in cursor.fetchall()}
print("Topics en el bag:", list(topics.keys()))

def read_float32(topic_name):
    tid = topics[topic_name]
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,)
    )
    times, vals = [], []
    for ts, data in cursor.fetchall():
        val = struct.unpack_from('<f', data, 4)[0]
        times.append(ts * 1e-9)  # nanosegundos → segundos
        vals.append(val)
    return np.array(times), np.array(vals)

ts_r, wr = read_float32('/VelocityEncR')
ts_l, wl = read_float32('/VelocityEncL')
conn.close()

# Detectar segmentos donde el robot se mueve (|wr| > umbral)
threshold = 0.5  # rad/s

def encontrar_segmentos(vals, umbral):
    moving = (np.abs(vals) > umbral).astype(int)
    diff = np.diff(np.concatenate([[0], moving, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

segmentos_r = encontrar_segmentos(wr, threshold)
segmentos_l = encontrar_segmentos(wl, threshold)

print(f"\nSegmentos detectados en wr: {len(segmentos_r)}")
print(f"Segmentos detectados en wl: {len(segmentos_l)}")

k_r_list = []
k_l_list = []

n = min(len(segmentos_r), len(segmentos_l), 3)
for i in range(n):
    sr, er = segmentos_r[i]
    sl, el = segmentos_l[i]

    wr_seg = wr[sr:er]
    wl_seg = wl[sl:el]

    mean_r = np.mean(np.abs(wr_seg))
    mean_l = np.mean(np.abs(wl_seg))
    std_r  = np.std(wr_seg)
    std_l  = np.std(wl_seg)

    k_r_i = std_r / mean_r
    k_l_i = std_l / mean_l

    k_r_list.append(k_r_i)
    k_l_list.append(k_l_i)

    print(f"\n--- Corrida {i+1} ---")
    print(f"  wr: mean={mean_r:.4f} rad/s  std={std_r:.4f}  k_r={k_r_i:.4f}")
    print(f"  wl: mean={mean_l:.4f} rad/s  std={std_l:.4f}  k_l={k_l_i:.4f}")

k_r_final = np.mean(k_r_list)
k_l_final = np.mean(k_l_list)

print(f"\n{'='*40}")
print(f"k_r final (promedio) = {k_r_final:.4f}")
print(f"k_l final (promedio) = {k_l_final:.4f}")
print(f"{'='*40}")
print("\nActualiza estos valores en:")
print("  localisation.py  → self.k_r y self.k_l")
print("  multi_puzzlebot_launch.py → parametros de r1_kinematic")
