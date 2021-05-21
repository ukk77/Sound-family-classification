res = model.predict(x_test_new)
arr = res.tolist()
result = []
result_prob = []
for row in arr:
  result.append(row.index(max(row)))
  result_prob.append(max(row))
print(result)
print(result_prob)
vals = [0] * 10 
done = []
for n in range(len(result)):
  if result_prob[n] > 0.9 or result_prob[n] < 0.1:
    if result[n] not in done:
      vals[result[n]] = n
    done.append(result[n])
print(vals)


from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import glob

content = []
target_classes = [ "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal" ]

for n in range(len(vals)):
  content.append(x_test[vals[n]])

for num in range(len(content)):  
  sh = np.array(content[num], dtype=float).shape[0]
  length = sh / 2000
  timen = np.linspace(0., length, sh)
  plt.plot(timen, content[num])
  plt.title("" + target_classes[num])
  plt.xlabel("Time [s]")
  plt.ylabel("Amplitude")
  plt.show();

