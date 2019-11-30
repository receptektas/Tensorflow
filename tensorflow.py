# -*- coding: utf-8 -*-

import numpy as np  # lineer algebra
import tensorflow as tf # ai
from tensorflow import keras # ai
import matplotlib.pyplot as plt # plot and graph

fashion_mnist = keras.datasets.fashion_mnist # loaded the data

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # this code return four diferent data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""# **Explore the data**
* Before training the model, let's examine the format of the data set.
* In the following section, there are 60,000 images in the training set, each shown as 28 x 28 pixels.
"""

train_images.shape # 28x28 in size 60000 picture

len(train_labels) # there is 60000 labels

train_labels # labels take a value between 0 and 9

test_images.shape # test set contains 10000 images with a size of 28x28

len(test_labels) # and the set contains 10000 labels

"""# **Preprocess the data**
* Now we must process the data
"""

plt.figure()
plt.imshow(train_images[0])
plt.colorbar() # for color scala
plt.grid(False) # squared ground
plt.show()

"""* Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
* To do so, divide the values by 255.
* It's important that the training set and the testing set be preprocessed in the same way:
"""

train_images = train_images / 255.0
test_images = test_images / 255.0

"""# Let's look at the top 25 data to check what we're doing."""

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([]) # space instead of pixels number
  plt.yticks([]) # space instead of pixels number
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary) # colorMap : default color scales binary : balack and white
  plt.xlabel(class_names[train_labels[i]]) # name of x axis 
plt.show()

"""# **Build the model**
* Building the neural network
* Configuring the layers of the model 
* Compiling the model.

##Set up the layers
@ The basic building block of a neural network is the layer. Layers extract representations from the data fed into them.

@ Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # 28x28=784 one dimensional array
    keras.layers.Dense(128, activation='relu'), # 128 neurons
    keras.layers.Dense(10, activation='softmax') 
])

"""## Compile the model
* Loss function : This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
* Optimizer —This is how the model is updated based on the data it sees and its loss function.
* Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
"""

model.compile(optimizer= 'adam', # default (0.1)
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""# **Train the model**
* Feed the created model with training sets (train_images , train_labels )
* The model learns to associate images and labels
* You ask the model to make predictions about a test set. (test_images , test_labels)
"""

model.fit(train_images , train_labels , epochs=10)

"""#**Evaluate accuracy**
* Next , Compare how the model performs on the test dataset.
"""

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy:',test_acc)

"""**Overfitting** is when a machine learning model performs worse on new, previously unseen inputs than on the training data.

*******************************************************************
Burada eğitim seti geçmiş 10 yılın soruları, model sınavın geçmiş senelere benzeyeceğini düşünmeniz, test seti hiç görmediğimiz istatistik sınavı, başarı kriteri aldığınız not. Sınav soruları beklendiğiniz gibi gelmez de kötü not alırsanız bu olaya overfitting denir.
Overfitting probleminde model çalıştığımız veri seti üzerinde harika sonuçlar verir (training error düşük) fakat hiç görmediği yeni veri setleri üzerinde başarısız tahminler yapar. (test error yüksek) Yukarıda örneğini verdiğim gibi bütün soruları ezberlemek modelimizi çok kompleks hale getiriyor ve gürültü (noise) barındırıyor. Biz eğitim setindeki değişkenlerin arasındaki gerçek ilişkiyi modellemeye çalışıyoruz sadece o veri setine özgü gürültüyü değil.

Underfitting (High Bias)
Hoca bu konuyu sormaz şu konuyu sormaz diye diye kafanıza göre konuları çalışmaktan vazgeçip düşük not alırsanız buna da underfitting denir.
Diğer bir deyişle eğer modelimizi eğitim (training) veri seti üzerinde çok basit olarak kurguladıysak hiç görmediğimiz test verisi üzerinde başarısız tahminler (sallama) yaparız ve gerçek değerle tahmin ettiğimiz değer arasındaki fark çok olur.![alt text](https://miro.medium.com/proxy/1*_7OPgojau8hkiPUiHoGK_w.png)
**************************************************************

#**Make predictions**
With the model trained, you can use it to make predictions about some images.
"""

predictions = model.predict(test_images)

"""Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction."""

predictions[0]

"""A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value."""

np.argmax(predictions[0])

test_labels[0]

def plotImage(i, predictionsArray, trueLabel, img):
  predictionsArray, trueLabel, img = predictionsArray, trueLabel[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predictedLabel = np.argmax(predictionsArray)
  if predictedLabel == trueLabel:
    color = 'green'
  else:
    color='red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predictedLabel],
                                       100*np.max(predictionsArray),
                                       class_names[trueLabel]),
                                       color = color)

def plotValueArray(i, predictionsArray, trueLabel):
  predictionsArray, trueLabel = predictionsArray, trueLabel[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisPlot = plt.bar(range(10), predictionsArray, color = '#777777')
  plt.ylim([0,1])
  predictedLabel = np.argmax(predictionsArray)

  thisPlot[predictedLabel].set_color('red')
  thisPlot[trueLabel].set_color('green')

"""Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are green and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label."""

i = 1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plotImage(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plotValueArray(i, predictions[i], test_labels)
plt.show()





















