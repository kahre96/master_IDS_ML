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
        self.datasize = data.shape[1] - 1  # amount of features
        self.datapoints = data.shape[0] - 1
        self.index = 0  # index of the current data point
        self.state = None
        self.penalty = -10
        self.reward = 1
        self.attack_reward = 100
        self.attack_penalty = -100

    def step(self, action):
        label = self.data.iloc[self.index, -1]  # get label from dataset
        # FIX LABEL STATE AND INDEXX!!!!
        if action == label:
            reward = self.reward  # match, give positive reward
            correct = 1
        else:
            correct = 0
            reward = self.penalty  # not matching, give negative reward

        correct_att = 0
        false_pos=0
        if label == 1:
            attack = 1
            if correct == 1:
                reward = self.attack_reward
                correct_att = 1
            else:
                reward = self.attack_penalty


        else:
            attack = 0
            if action == 1:
                false_pos += 1
        self.index = random.randint(0, self.datapoints)

        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.datasize)
        return self.state, reward, correct, attack, correct_att, false_pos

    def reset(self):
        self.index = random.randint(0, self.datapoints)
        self.state = self.data.iloc[self.index, :-1].values.reshape(1, self.datasize)
        return self.state



    def set_reward(self, reward,penalty):
        self.reward = reward
        self.penalty = penalty

    def set_atk_reward(self, a_reward,a_penalty):
        self.attack_reward = a_reward
        self.attack_penalty = a_penalty

    def render(self, mode='human', close=False):
        pass

