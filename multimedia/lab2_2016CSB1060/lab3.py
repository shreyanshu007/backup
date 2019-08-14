import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
import sounddevice as sd
import soundfile as sf



def q2():
	Fs = 100
	for f in range(10):
		x = np . arange ( Fs )
		y = np.sin((2*np.pi*f*x*10)/Fs)
		plt.plot((np.fft.fftfreq(Fs, 1.0/Fs)), np.abs(fft(y)))
		plt.savefig('sinfft')
		plt.close()


def q3():
	filename = 'killbill.wav'
	data, fs = sf.read(filename, dtype='float32')
	y = []
	for i in data:
	    y.append(i[0])
	x = np.arange(len(y))
	plt.plot(np.abs(np.fft.fftfreq(len(y), 1.0/len(y))), fft(y))
	plt.savefig('killbill')
	plt.close()



def q4():
	filename = 'male.wav'
	data, fs = sf.read(filename, dtype='float32')
	y = []
	for i in data:
	    y.append(i[0])
	x = np.arange(len(y))
	plt.plot(np.abs(np.fft.fftfreq(len(y), 1.0/len(y))), fft(y))
	plt.savefig('male')
	plt.close()

	# -----------------

	filename = 'female.wav'
	data, fs = sf.read(filename, dtype='float32')

	y = []
	for i in data:
	    y.append(i[0])
	x = np.arange(len(y))
	plt.plot(np.abs(np.fft.fftfreq(len(y), 1.0/len(y))), fft(y))
	plt.savefig('female')
	plt.close()


def q5():
	filename = 'male.wav'
	data, fs = sf.read(filename, dtype='float32')
	y = []
	for i in data:
	    y.append(i[0])
	x = np.arange(len(y))
	plt.plot(np.abs(np.fft.fftfreq(len(y), 1.0/len(y))), fft(y))
	plt.savefig('speech')
	plt.close()

	# -----------------
	print("wait for it")
	filename = 'music.wav'
	data, fs = sf.read(filename, dtype='float32')
	y = []
	for i in data:
	    y.append(i[0])
	x = np.arange(len(y))
	plt.plot(np.abs(np.fft.fftfreq(len(y), 1.0/len(y))), fft(y))
	plt.savefig('music')
	plt.close()



q2()
q3()
q4()
q5()