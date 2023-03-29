import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import numpy as np
class DQNAgent:
    def __init__(self, state_size, action_size, layer_amount, neurons, epsilon, decay, learn_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = decay
        self.learning_rate = learn_rate
        self.model = self._build_model(layer_amount, neurons)


    def _build_model(self, layer_amount, neurons):
        model = keras.Sequential()
        model.add(layers.Dense(neurons, input_dim=self.state_size, activation='relu'))
        for i in range(layer_amount):
            model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear')) # sigmoid??
        model.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=["binary_accuracy"])
        return model

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,1)
        tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        return np.argmax(self.model.predict(tensor, verbose=0, use_multiprocessing=True)[0])  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward in minibatch:
            target = reward + self.gamma * tf.reduce_max(self.model(state)[0])
            target_f = self.model(state)
            target_f = tf.tensor_scatter_nd_update(target_f, [[0, action]], [target])
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def update_explore_rate(self, epsilon):
        self.epsilon = epsilon

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)