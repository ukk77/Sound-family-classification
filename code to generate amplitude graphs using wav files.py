from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import glob

def read_f():
  files = []
  for file in glob.glob("/content/*.wav"):
    files.append(file)
  return files

def main():
  files = read_f()
  content = []

  for f in files:  
    print("-----")
    print(f)  
    audio = read(f)
    content.append(audio)
    sh = np.array(audio[1], dtype=float).shape[0]
    length = sh / audio[0]
    timen = np.linspace(0., length, sh)
    plt.plot(timen, audio[1])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show();

main()