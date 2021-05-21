import matplotlib.pyplot as plt

acc_vals = history.history['accuracy']
epochs = range(1, len(acc_vals)+1)
plt.plot(epochs, acc_vals, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#history = model.fit(.....)