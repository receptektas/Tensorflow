# -*- coding: utf-8 -*-
"""1textClassificationWithTensorflowHub.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sRl6J2ix8fwYo2xDjmvCRXO6LFknSbTM

#**Text classification with TensorFlow Hub: Movie reviews**

####**Firstly import the necessary libraries and check the libraries**
"""

import tensorflow as tf
if (tf.config.experimental.list_physical_devices("GPU") == [] or tf.__version__ != '2.0.0'):
  !pip install tensorflow-gpu
  print("** Tensorflow updated **")
import numpy as np
!pip install -q tensorflow-hub
!pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version : ",tf.__version__)
print("Eager mode : ",tf.executing_eagerly())
print("Hub Version : ",hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

"""####**Download the IMDB dataset**

* Split the training set into 60% and 40%, so we'll end up with 15,000 examples
 for training, 10,000 examples for validation and 25,000 examples for testing.
"""

trainValidationSplit = tfds.Split.TRAIN.subsplit([6,4])
(trainData, validationData), testData = tfds.load(
    name='imdb_reviews',
    split=(trainValidationSplit, tfds.Split.TEST),
    as_supervised = True)

"""#**Explore the data**
* Let's take a moment to understand the format of the data. 
* Each example is a sentence representing the movie review and a corresponding label. 
* The sentence is not preprocessed in any way. 
* The label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

Let's print first 10 examples.
"""

trainExamplesBatch, trainLabelsBatch = next(iter(trainData.batch(10)))
trainExamplesBatch

"""####Let's print the top ten labels"""

trainLabelsBatch

"""#**Build the model**

* In this example, the input data consists of sentences. 
The labels to predict are either 0 or 1.


> One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have three advantages :
1. We don't have to worry about text preprocessing
2. We can benefit from transfer learning
3. The embedding has a fixed size, so it's simpler to process

For this example we will use a pre-trained text embedding model from TensorFlow Hub called " https://google/tf2-preview/gnews-swivel-20dim/1 "

Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, and try it out on a couple of input examples. Note that no matter the length of the input text, the output shape of the embeddings is: **(num_examples, embedding_dimension).**
"""

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hubLayer = hub.KerasLayer(embedding, input_shape=[],
                          dtype=tf.string, trainable=True)
hubLayer(trainExamplesBatch[:3])

"""####Now let's create the full model"""

model = tf.keras.Sequential()
model.add(hubLayer)                                         # https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))   # sigmoid : binary classification 
model.summary()

"""###Loss function and optimizer"""

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # mean_squared_error but not better than this
              metrics=['accuracy'])

"""#**Train the model**
* Train the model for 20 epochs in mini-batches of 512 samples.
* During training, monitor the loss and accuracy of the model in 10,000 samples from the validation set
"""

history = model.fit(trainData.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data= validationData.batch(512),
                    verbose=1)

"""#**Evaluate the model**
And let's see how the model performs. Two values will be returned. Loss and accuracy.
"""

result = model.evaluate(testData.batch(512), verbose=1)
for name, value in zip(model.metrics_names, result):
  print("%s: %.3f" % (name, value))

"""#**The End**
>**See you next coding**
"""