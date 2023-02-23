import pyarrow.feather as feather
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from memory_profiler import profile
from pympler import asizeof
import gc

def run():
    action_size = 2
    #s_clean_CIC2017
    #"bin_all_featuresCIC2017.feather"
    #

    df = pd.read_feather("binary_31features_CIC2017.feather")
    env = MyEnv(df)
    del df
    gc.collect()
    # EPISODES_array = [2000]
    batch_size = 320
    # layerarray = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # epsilon_array = [0.995, 0.99, 0.95, 0.90]
    # epsilon_decay_array = [0.995, 0.99, 0.95, 0.90]
    epsilon = 0.995
    # epsilon_decay_array = [0.99]
    penalty = 0
    learn_rate = 0.001
    layer_amount = 3
    neurons_amount = 64
    EPISODES = 2000
    decay = 0.996

    train_model(layer_amount, neurons_amount, EPISODES, batch_size, epsilon, decay, env, action_size, penalty, learn_rate)




class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        self.action_space = spaces.Discrete(2)  # 2 possible actions
        # self.observation_space = spaces.Discrete(1) # only one observation needed
        # self.reward_range = (-1, 1) # rewards range from -1 to 1
        self.data = data  # dataset
        self.datasize = data.shape[1] - 1  # amount of features
        self.datapoints = data.shape[0] - 1
        self.index = 0  # index of the current data point
        self.state = None
        self.penalty = 0

    def step(self, action):
        label = self.data.iloc[self.index, -1]  # get label from dataset
        if action == label:
            reward = 1  # match, give positive reward
            correct = 1
        else:
            correct = 0
            reward = self.penalty  # not matching, give negative reward
        self.index = random.randint(0, self.datapoints)  # move to next data point
        # row = self.data.get_chunk(1)
        # if row.empty:
        #    self.done = True
        #    return 0,0,self.done,{}
        # self.state= row.iloc[:,1:-1].values
        # self.label = row.iloc[:,-1:].values

        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.datasize)
        return self.state, reward, correct

    def reset(self):
        self.index = random.randint(0, self.datapoints)
        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.datasize)
        return self.state



    def set_penalty(self, penalty):
        self.penalty = penalty

    def render(self, mode='human', close=False):
        pass






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
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=["binary_accuracy"])
        return model

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,1)
        tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        gc.collect()
        return np.argmax(np.argmax(self.model(tensor)[0]))  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward in minibatch:
            target = reward + self.gamma * tf.reduce_max(self.model(state)[0])
            target_f = self.model(state)
            target_f = tf.tensor_scatter_nd_update(target_f, [[0, action]], [target])
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)


def train_model(layer_amount, neurons, EPISODES, batch_size, epsilon, decay, env, action_size, penalty, learn_rate):

    env.set_penalty(penalty)
    progress = np.zeros(EPISODES)
    start_time = time.time()
    agent = DQNAgent(env.datasize, action_size, layer_amount - 1, neurons, epsilon, decay, learn_rate)
    for e in range(EPISODES):
        episode_correct = 0
        state = env.reset()
        for index in range(batch_size):
            action = agent.act(state)
            next_state, reward, correct = env.step(action)
            agent.remember(state, action, reward)
            state = next_state
            episode_correct += correct


        agent.replay(32)


        # print(f"batch: {batch}/{batch_amounts} acc: {batch_reward/train_batch}")
        acc = episode_correct / batch_size
        print(f"episode: {e}/{EPISODES}, e: {agent.epsilon:.2f}, total acc: {acc} ")
        progress[e] = acc
        del acc
        del state
        del episode_correct

        gc.collect()
    # with open('gridresults.txt', 'a') as f:
    #       f.write(str(tot_reward))
    traintime = time.time() - start_time

    data = pd.read_feather("features_results.feather")
    data.loc[len(data)] = [epsilon,decay,layer_amount, neurons, progress, traintime,EPISODES, agent.epsilon, penalty, learn_rate, env.datasize, "none"]
    data.to_feather("features_results.feather")
    #data = pd.read_feather("full_epsilonresults.feather")
    #data.loc[len(data)] = [epsilon,decay,layer_amount, neurons, progress, traintime,EPISODES, agent.epsilon]
    #data.to_feather("full_epsilonresults.feather")

    #agent.save(f"{epsilon}_{decay}_{layer_amount}x{neurons}_{EPISODES}_ee{agent.epsilon}.h5")
    # data = pd.read_feather("gridresults.feather")
    # data.loc[len(data)] = [layer_amount, neurons, progress, traintime]
    # data.to_feather("gridresults.feather")
    del agent
    del progress
    del start_time
    del data
    del traintime
    gc.collect()






if __name__ == "__main__":

    run()

