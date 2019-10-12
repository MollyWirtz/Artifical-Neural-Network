
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


# Upload data files.
images = numpy.load('images.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
labels = numpy.load('labels.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

# Flatten provided data.
flattened_imgs = []
for img in images:
    img = img.flatten()
    flattened_imgs.append(img)
one_hot_lbls = []
for lbl in labels:
    new = []
    for i in range(0, 10):
        if i == lbl:
            new.append(1)
        else:
            new.append(0)
    one_hot_lbls.append(new)
    new.clear

new_lbls = to_categorical(labels)


# Build Artifical Neural Network with Keras
shape_imgs =  flattened_imgs[0].shape
shape_lbls = (10,) #one_hot_lbls[0].shape
sgd = optimizers.SGD(lr=0.001,momentum=0.0, decay=0.0, nesterov=False)

model = Sequential()
model.add(Dropout(rate = 0.2, input_shape=shape_imgs))  # Dropout on visible layer
for i in range (0, 10):                                       # 10 hidden layers
    model.add(Dense(units=80,activation='relu'))
model.add(Dense(units=10, activation='softmax'))              # Output layer

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'mse'])


# k-fold 1
# test = 0-20
test_set_range = int(len(one_hot_lbls)*0.20)
test_set_imgs = numpy.array(flattened_imgs[0:test_set_range])
test_set_lbls = numpy.array(one_hot_lbls[0:test_set_range])
# validation = 20-40
valid_set_range = int(len(one_hot_lbls)*0.20)
valid_set_imgs = numpy.array(flattened_imgs[test_set_range:(valid_set_range+test_set_range)])
valid_set_lbls = numpy.array(one_hot_lbls[test_set_range:(valid_set_range+test_set_range)])
# train = 40-100
train_set_range = int(len(one_hot_lbls)*0.60)
train_set_imgs= numpy.array(flattened_imgs[valid_set_range:(train_set_range+valid_set_range)])
train_set_lbls = numpy.array(one_hot_lbls[valid_set_range:(train_set_range+valid_set_range)])

# Iterate over training data K-fold 1
training_data = model.fit(train_set_imgs, train_set_lbls, epochs=500, batch_size=512, validation_data=(valid_set_imgs,valid_set_lbls))

# Plotting Training and Validation Accuracy
# get_ipython().run_line_magic('matplotlib', 'inline')
x = [i for i in range(1, 501)] #Epochs
y1 = training_data.history['acc']
y2 = training_data.history['val_acc']

# plt.plot(x, y1, 'r-')
# plt.plot(x, y2, 'b-')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Task Two Training and Validation Accuracy with Dropout K-Fold One')
# plt.legend(['Training Accuracy', 'Validation Accuracy'])
# plt.savefig('AccPlotFoldOne.png')
# plt.show()
#
# Evaluate performance on test data
loss_and_metrics = model.evaluate(test_set_imgs, test_set_lbls, batch_size=512)
print(model.metrics_names)
print(loss_and_metrics)

# Confusion Matrix
prediction = model.predict(test_set_imgs)
cm = confusion_matrix(numpy.argmax(test_set_lbls, axis=1), numpy.argmax(prediction, axis=1))
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.matshow(cm, cmap=mpl.cm.binary)


# k-fold 2
# train = 0-60
train_set_range = int(len(one_hot_lbls)*0.60)
train_set_imgs= numpy.array(flattened_imgs[valid_set_range:(train_set_range+valid_set_range)])
train_set_lbls = numpy.array(one_hot_lbls[valid_set_range:(train_set_range+valid_set_range)])
# test = 60-80
test_set_range = int(len(one_hot_lbls)*0.20)
test_set_imgs = numpy.array(flattened_imgs[0:test_set_range])
test_set_lbls = numpy.array(one_hot_lbls[0:test_set_range])
# validation = 80-100
valid_set_range = int(len(one_hot_lbls)*0.20)
valid_set_imgs = numpy.array(flattened_imgs[test_set_range:(valid_set_range+test_set_range)])
valid_set_lbls = numpy.array(one_hot_lbls[test_set_range:(valid_set_range+test_set_range)])

# Iterate over training data K-fold 2
training_data = model.fit(train_set_imgs, train_set_lbls, epochs=500, batch_size=512, validation_data=(valid_set_imgs,valid_set_lbls))

# Plotting Training and Validation Accuracy
# get_ipython().run_line_magic('matplotlib', 'inline')
x = [i for i in range(1, 501)] #Epochs
y1 = training_data.history['acc']
y2 = training_data.history['val_acc']

# plt.plot(x, y1, 'r-')
# plt.plot(x, y2, 'b-')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Task Two Training and Validation Accuracy with Dropout K-Fold Two')
# plt.legend(['Training Accuracy', 'Validation Accuracy'])
# plt.savefig('AccPlotFoldTwo.png')
# plt.show()
#
# Evaluate performance on test data
loss_and_metrics = model.evaluate(test_set_imgs, test_set_lbls, batch_size=512)
print(model.metrics_names)
print(loss_and_metrics)

# Confusion Matrix
prediction = model.predict(test_set_imgs)
cm = confusion_matrix(numpy.argmax(test_set_lbls, axis=1), numpy.argmax(prediction, axis=1))
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.matshow(cm, cmap=mpl.cm.binary)


# k-fold 3
# validation = 0-20
valid_set_range = int(len(one_hot_lbls)*0.20)
valid_set_imgs = numpy.array(flattened_imgs[test_set_range:(valid_set_range+test_set_range)])
valid_set_lbls = numpy.array(one_hot_lbls[test_set_range:(valid_set_range+test_set_range)])
# test = 20-40
test_set_range = int(len(one_hot_lbls)*0.20)
test_set_imgs = numpy.array(flattened_imgs[0:test_set_range])
test_set_lbls = numpy.array(one_hot_lbls[0:test_set_range])
# train = 40-100
train_set_range = int(len(one_hot_lbls)*0.60)
train_set_imgs= numpy.array(flattened_imgs[valid_set_range:(train_set_range+valid_set_range)])
train_set_lbls = numpy.array(one_hot_lbls[valid_set_range:(train_set_range+valid_set_range)])

# Iterate over training data K-fold 3
training_data = model.fit(train_set_imgs, train_set_lbls, epochs=500, batch_size=512, validation_data=(valid_set_imgs,valid_set_lbls))

# Plotting Training and Validation Accuracy
# get_ipython().run_line_magic('matplotlib', 'inline')
x = [i for i in range(1, 501)] #Epochs
y1 = training_data.history['acc']
y2 = training_data.history['val_acc']

# plt.plot(x, y1, 'r-')
# plt.plot(x, y2, 'b-')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Task Two Training and Validation Accuracy with Dropout Fold Three')
# plt.legend(['Training Accuracy', 'Validation Accuracy'])
# plt.savefig('AccPlotFoldThree.png')
# plt.show()
#
# Evaluate performance on test data
loss_and_metrics = model.evaluate(test_set_imgs, test_set_lbls, batch_size=512)
print(model.metrics_names)
print(loss_and_metrics)

# Confusion Matrix
prediction = model.predict(test_set_imgs)
cm = confusion_matrix(numpy.argmax(test_set_lbls, axis=1), numpy.argmax(prediction, axis=1))
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.matshow(cm, cmap=mpl.cm.binary)

