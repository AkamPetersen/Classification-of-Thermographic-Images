import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.utils import normalize, to_categorical
from time import time

IMAGE_H = 224
IMAGE_W = 224
BATCH_SIZE = 16
EPOCHS = 20
learning_rate = 0.0001


x_train = np.load("train.npy", allow_pickle=True)
y_train = np.load("train_labels.npy", allow_pickle=True)

x_val = np.load("val.npy", allow_pickle=True)
y_val = np.load("val_labels.npy", allow_pickle=True)

x_test = np.load("test.npy", allow_pickle=True)
y_test = np.load("test_labels.npy", allow_pickle=True)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0


print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

print(x_test.shape)
print(y_test.shape)

activation = 'relu'
model = Sequential()
model.add(Conv2D(128, 3, activation=activation, padding='same', input_shape=(IMAGE_H, IMAGE_W, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(160, 3, activation=activation, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(160, 3, activation=activation, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, activation=activation, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, 3, activation=activation, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(384, activation=activation))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

start = time()

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val), shuffle=True)

print(f"Training completed in: {(time()-start):.2f}  S")

model.save('CNN_LeavesClassificationTuned_16.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs2 = range(1, len(acc) + 1)
plt.plot(epochs2, acc, 'y', label='Training acc')
plt.plot(epochs2, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("ACC_plot")
plt.show()
plt.close()


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs1 = range(1, len(loss) + 1)
plt.plot(epochs1, loss, 'y', label='Training loss')
plt.plot(epochs1, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Loss_plot")
plt.show()
plt.close()

model.evaluate(x_test, y_test, verbose=1)

classes1 = ["Blast", "BLB", "Healthy", "Hispa", "Leaf Folder", "Leaf Spot"]
classes1_reshape = np.reshape(classes1, (6, 1))
classes2 = [0, 1, 2, 3, 4, 5]

y_prediction = model.predict(x_test)
y_prediction = np.argmax(y_prediction, axis=1)
y_test = np.argmax(y_test, axis=1)
result = confusion_matrix(y_test, y_prediction, labels=classes2)
display_c_m = ConfusionMatrixDisplay(result, display_labels=classes1)
display_c_m.plot(cmap='Blues')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Confusion Matrix', fontsize=14)
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()

print(classification_report(y_test, y_prediction, target_names=classes1))

FP = result.sum(axis=0) - np.diag(result)
FN = result.sum(axis=1) - np.diag(result)
TP = np.diag(result)
TN = result.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
TPR = np.reshape(TPR, (6, 1))
#print(TPR)
# Specificity or true negative rate
TNR = TN/(TN+FP)
TNR = np.reshape(TNR, (6, 1))
#print(TNR)
# Precision or positive predictive value
PPV = TP/(TP+FP)
PPV = np.reshape(PPV, (6, 1))
#print(PPV)
# Negative predictive value
NPV = TN/(TN+FN)
NPV = np.reshape(NPV, (6, 1))
#print(NPV)
# Fall out or false positive rate
FPR = FP/(FP+TN)
FPR = np.reshape(FPR, (6, 1))
#print(FPR)
# False negative rate
FNR = FN/(TP+FN)
FNR = np.reshape(FNR, (6, 1))
#print(FNR)
# False discovery rate
FDR = FP/(TP+FP)
FDR = np.reshape(FDR, (6, 1))
#print(FDR)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)
ACC = np.reshape(ACC, (6, 1))
#print(ACC)

res = np.concatenate((classes1_reshape, TPR, TNR, ACC), axis=1)

df = pd.DataFrame(res, columns=["Class", "Sensitivity", "Specificity", "Acc"])
print(df)