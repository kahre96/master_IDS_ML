from DQN import DQNAgent
import tensorflow as tf
import random
from collections import deque



class DDQNAgent(DQNAgent):

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
        self.target_model = self._build_model(layer_amount, neurons)
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward in minibatch:
            target = reward + self.gamma * tf.reduce_max(self.target_model(state)[0])
            target_f = self.model(state)
            target_f = tf.tensor_scatter_nd_update(target_f, [[0, action]], [target])
            self.model.fit(state, target_f, epochs=1, verbose=0)


        if len(self.memory) % 10 == 0:
            self.target_model.set_weights(self.model.get_weights())

        del minibatch


    def load(self, name):
        self.model = tf.keras.models.load_model(name)
        self.target_model = tf.keras.models.load_model(name)
