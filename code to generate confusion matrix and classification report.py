from sklearn.metrics import confusion_matrix, classification_report

y_predicted = model.predict(x_test_new)
y_pred = np.argmax(y_predicted, axis=1)

y = []
for row in y_test:
  y.append(row.index(max(row)))

target_classes = [ "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal" ]

print(confusion_matrix(np.asarray(y),y_pred), )

print(classification_report(np.asarray(y),y_pred,target_names = target_classes ))
