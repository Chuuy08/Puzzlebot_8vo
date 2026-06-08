import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import os
import time

WORDS = ["stop"]
FS_RECORD = 16000
FS_MODEL  = 16000
DURATION  = 2
# Carpeta de salida: por defecto el directorio actual ("./recordings/<palabra>/").
# Se puede sobreescribir con la variable de entorno VOICE_RECORD_DIR, p.ej. para
# grabar directamente sobre el dataset fuente del paquete antes de reentrenar.
OUTPUT_DIR = os.environ.get("VOICE_RECORD_DIR", os.path.join(os.getcwd(), "recordings"))
DEVICE    = 9


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== GRABADOR DE PALABRAS ===")
    print(f"Guardando en: {OUTPUT_DIR}")

    for word in WORDS:
        word_dir = os.path.join(OUTPUT_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        input(f"--- Palabra: '{word.upper()}' --- Presiona ENTER cuando estés listo...")

        for i in range(1, 17):
            print(f"  Grabación {i} — Di '{word}' en 3...")
            time.sleep(0.6)
            print("  2...")
            time.sleep(0.6)
            print("  1...")
            time.sleep(0.6)
            print("  ¡HABLA!")

            audio = sd.rec(int(DURATION * FS_RECORD), samplerate=FS_RECORD,
                   channels=2, device=DEVICE, dtype='float32')
            sd.wait()
            audio = audio.mean(axis=1)


            # Filtro pasa-altas
            #b, a = sig.butter(4, 80, btype='high', fs=FS_RECORD)
            #audio = sig.filtfilt(b, a, audio)

            # Resample 48000 -> 16000
            audio_16k = audio
            audio_16k = audio_16k / (np.max(np.abs(audio_16k)) + 1e-10)
            print(f"  Audio float32 max: {audio_16k.max():.4f}, min: {audio_16k.min():.4f}")

            # Convertir a int16 para guardar
            audio_int16 = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
            filename = os.path.join(word_dir, f"pipe_{i:02d}.wav")
            wav.write(filename, FS_MODEL, audio_int16)
            print(f"  Guardado: {filename}")

            """for j in range(2, 16):
                noise = np.random.normal(0, 0.005, audio_16k.shape)
                audio_aug = (audio_16k + noise).clip(-1, 1)
                audio_aug_int16 = (audio_aug * 32768).clip(-32768, 32767).astype(np.int16)
                filename_aug = os.path.join(word_dir, f"pipe_{j:02d}.wav")
                wav.write(filename_aug, FS_MODEL, audio_aug_int16)
                print(f"  Copia generada: {filename_aug}")"""

            print()

    print("✅ Listo!")


if __name__ == '__main__':
    main()
