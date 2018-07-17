import numpy as np
import librosa

imi_arr = np.zeros(50)
ref_arr = np.zeros(50)

sr = 44100


librosa.output.write_wav('imi_silence.wav', imi_arr, sr)
librosa.output.write_wav('ref_silence.wav', ref_arr, sr)


