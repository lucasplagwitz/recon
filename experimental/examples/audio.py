from pydub import AudioSegment
from pydub.playback import play

import matplotlib.pyplot as plt
from recon.interfaces import Smoothing, SmoothBregman

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


audio = read('./data/09.m4a')
audio = read('./data/08.m4a')
audio = read('./data/07.mp3')


#samples = audio.get_array_of_samples()
samples = audio[1] #[:,0]


samples = samples #[:10*(len(samples))//100, :]


alpha = np.ones(samples.shape)*0.001
fac = np.max(samples)

samples = (samples/fac) #+ np.random.normal(0, 0.015, size=np.shape(samples))

#tv = Smoothing(domain_shape=samples.shape, reg_mode='tv', alpha=0.6, tau='calc')
#x = tv.solve(samples, tol=10**(-5))

tik = Smoothing(domain_shape=samples.shape, reg_mode='tikhonov', alpha=6, tau='calc') #1.4
x = tik.solve(samples, tol=10**(-5))


#bregman = SmoothBregman(domain_shape=samples.shape,
#                        reg_mode='tv',
#                        alpha=0.1,
#                        tau='calc',
#                        assessment=0.01*np.sqrt(np.prod(samples.shape)))
#x = bregman.solve(samples, tol=10**(-5))


x = x*fac
samples = samples*fac
print("test")



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.plot(list(range(len(samples))), list(samples))
ax2.plot(list(range(len(samples))), list(x))
ax3.plot(list(range(len(samples))), list(samples-x))
plt.show()

samples = samples.astype(np.int16)
x = x.astype(np.int16)

error = (samples - x).astype(np.int16)

error_audio = AudioSegment(
    (error).tobytes(),
    frame_rate=44100,
    sample_width=error.dtype.itemsize,
    channels=2
)

reconstruction = AudioSegment(
    x.tobytes(),
    frame_rate=44100,
    sample_width=x.dtype.itemsize,
    channels=2
)

plain = AudioSegment(
    samples.tobytes(),
    frame_rate=44100,
    sample_width=samples.dtype.itemsize,
    channels=2
)

# test that it sounds right (requires ffplay, or pyaudio):
play(reconstruction)
play(plain)
play(error_audio)

reconstruction.export("./data/short.wav", format="wav")

print("test")