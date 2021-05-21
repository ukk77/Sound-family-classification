from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import glob


res = model.predict(x_test_new)
arr = res.tolist()
result = []
result_prob = []
vals = [0] * 10 
done = []
count = 0

for row in arr:
  result.append(row.index(max(row)))
  result_prob.append(max(row))
  row.sort()
  if row[-1] - row[-2] <= 10:
    if result[-1] not in done:
      vals[result[-1]] = count 
  done.append(result[-1])
  count += 1
  
# print(vals)

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
