import librosa
import glob
import time
DATA_DIR = "./MUSIC_DATA/maestro-v3.0.0/2004"
start = time.perf_counter()
f = librosa.load(f"{DATA_DIR}/MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_01_Track01_wav.wav")
print("time", time.perf_counter() - start)
start = time.perf_counter()
librosa.load(f"{DATA_DIR}/MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_06_Track06_wav.wav")
print("time", time.perf_counter() - start)
start = time.perf_counter()
librosa.load(f"{DATA_DIR}/MIDI-Unprocessed_XP_06_R1_2004_01_ORIG_MID--AUDIO_06_R1_2004_01_Track01_wav.wav")
print("time", time.perf_counter() - start)

