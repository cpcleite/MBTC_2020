# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:41:36 2020

@author: cpcle
"""

from keras.models import Sequential
from keras.layers import Dense

# model
model = Sequential()

# Input and 1 hidden layer
model.add(Dense(4, input_shape=(2,),
                activation='tanh'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer = 'adam',
              loss='binary_crossentropy')

# Train model
history = model.fit(features, y_train, epochs=20,
            metrics='f1_weighted')

# Predict
y_pred = model.predict(features)