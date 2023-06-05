import gymnasium as gym
from gymnasium import spaces
import random
class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        self.action_space = spaces.Discrete(2)  # 2 possible actions
        # self.observation_space = spaces.Discrete(1) # only one observation needed
        # self.reward_range = (-1, 1) # rewards range from -1 to 1
        #self.data0 = data[data[' Label'] == 0]  # dataset
        #self.data1 = data[data[' Label'] == 1]
        self.data = data
        self.features = data.shape[1] - 1  # amount of features
        self.datapoints = data.shape[0] - 1
        self.index = 0  # index of the current data point
        self.state = None
        self.penalty = -10
        self.reward = 1
        self.attack_reward = 100
        self.attack_penalty = {1: -25,
                               2: -25,
                               3: -25,
                               4: -25,
                               5: -25,
                               6: -25,
                               7: -25,
                               8: -25,
                               9: -25,
                               10: -25,
                               11: -25,
                               12: -25,
                               13: -25,
                               14: -25
                               }

    def step(self, action):
        label = self.data.iloc[self.index, -1]  # get label from dataset

        if (action == 0 and label == 0) or (action != 0 and label != 0):
            reward = self.reward  # match, give positive reward
            correct = 1
        else:
            correct = 0
            reward = self.penalty  # not matching, give negative reward

        correct_att = 0
        false_pos=0
        if label != 0:
            attack = 1
            if correct == 1:
                reward = self.attack_reward
                correct_att = 1
            else:
                reward = self.attack_penalty
                #reward = self.attack_penalty[label]
        else:
            attack = 0
            if action != 0:
                false_pos += 1
        self.index = random.randint(0, self.datapoints)

        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.features)
        return self.state, reward, correct, attack, correct_att, false_pos, label

    def reset(self):
        self.index = random.randint(0, self.datapoints)
        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.features)
        return self.state



    def set_reward(self, reward,penalty):
        self.reward = reward
        self.penalty = penalty

    def set_atk_reward(self, a_reward,a_penalty):
        self.attack_reward = a_reward
        self.attack_penalty = a_penalty

    def render(self, mode='human', close=False):
        pass

