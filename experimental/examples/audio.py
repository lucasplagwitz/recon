from pydub import AudioSegment
from pydub.playback import play

import numpy as np
import matplotlib.pyplot as plt
from recon.interfaces import Smooth, SmoothBregman
from experimental.interfaces.satv import SATV

import pydub
import numpy as np

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_file(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


audio = read('/Users/lucasplagwitz/Desktop/test_chopin.m4a')

#samples = audio.get_array_of_samples()
samples = audio[1][:,0]

# only error
error_example = samples[int(0.3*(len(samples))//200):int(1.2*(len(samples))//200)].astype(np.int16)
ee_audio = AudioSegment(
    error_example.tobytes(),
    frame_rate=44100,
    sample_width=error_example.dtype.itemsize,
    channels=1
)
#play(ee_audio)

samples = samples[:15*(len(samples))//100]
#samples = error_example
#fac = np.max(samples)
#sop = Smooth(domain_shape=samples.shape, reg_mode='tikhonov', alpha=10.5, tau=0.00035)
#x = sop.solve(samples/fac, max_iter=400, tol=10**(-6))
#x = x*fac

alpha = np.ones(samples.shape)*0.001
fac = np.max(samples)
#sop = SATV(domain_shape=samples.shape,
#           reg_mode='tv',
#           alpha=alpha,
#           tau=0.035,
#           assessment=0.06*np.sqrt(samples.shape[0]),
#           noise_sigma=0.06)
#x = sop.solve(samples/fac, tol=10**(-5))

breg = SmoothBregman(domain_shape=samples.shape, reg_mode='tikhonov', alpha=1.1, tau=0.035, assessment=16)
x = breg.solve(samples/fac, tol=10**(-5))


x = x*fac

print("test")



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.plot(list(range(len(samples))), list(samples))
ax2.plot(list(range(len(samples))), list(x))
ax3.plot(list(range(len(samples))), list(samples-x))
plt.show()

x = x.astype(np.int16)

error = (samples -x).astype(np.int16)

error_audio = AudioSegment(
    (error).tobytes(),
    frame_rate=44100,
    sample_width=error.dtype.itemsize,
    channels=1
)

audio_segment = AudioSegment(
    x.tobytes(),
    frame_rate=44100,
    sample_width=x.dtype.itemsize,
    channels=1
)

audio = AudioSegment(
    samples.tobytes(),
    frame_rate=44100,
    sample_width=samples.dtype.itemsize,
    channels=1
)

# test that it sounds right (requires ffplay, or pyaudio):
play(error_audio)
play(audio)
play(audio_segment)

audio_segment.export("/Users/lucasplagwitz/Desktop/short.wav", format="wav")

print("test")