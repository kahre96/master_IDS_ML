import pyarrow.feather as feather
import pandas as pd
import numpy as np
from RL_IDS_env import MyEnv
from DQN import DQNAgent
from DDQN import DDQNAgent
from PPO2 import PPOAgent
from TD3 import TD3Agent
import time
import random

import tensorflow as tf

import gc
import math

random.seed(1337)

def run():
    action_size = 2
    #s_clean_CIC2017
    #"bin_all_featuresCIC2017.feather"
    # "norm_bin_allfeat_CIC2017.feather"
    # norm_bin_31feat_CIC2017
    #norm_6feat_CIC2017
    # "zsore3_norm_bin_69feat_CIC2017.feather


    #change df
    #change save
    #change data
    savefile= "results/TD3_results.feather"
    df = pd.read_feather("dataset/norm_bin_69feat_CIC2017.feather")
    env = MyEnv(df)
    del df
    gc.collect()
    batch_size = 320
    epsilon = 0.999
    penalty = -10
    reward = 1
    learn_rate = 0.001
    layer_amount = 3
    neurons_amount = 16
    EPISODES = 2000
    clip = 0.99 #0.99     0.2 0.4 0.6 0.8 0.9 0.99
    crit_discount= 0.95 #0.5
    decay = 0.996
    atk_reward = 2  #1 / 2
    atk_penalty = -25# -25 DQN -75 PPO

    env.set_reward(reward,penalty)
    env.set_atk_reward(atk_reward,atk_penalty)

    train_model(layer_amount, neurons_amount, EPISODES, batch_size, epsilon, decay, env,action_size,
                 learn_rate, savefile, reward,penalty,atk_reward, atk_penalty,clip,crit_discount)


def train_model(layer_amount, neurons, EPISODES, batch_size, epsilon, decay, env, action_size,
                learn_rate, savefile, the_reward, the_penalty,atk_reward,atk_penalty, clip, crit_discount):


    progress = np.zeros(EPISODES)
    precision_progress= np.zeros(EPISODES)
    recall_progress= np.zeros(EPISODES)
    f1_progress = np.zeros(EPISODES)

    agent = TD3Agent(env.datasize,action_size,layer_amount-1,neurons,2000,1,0.1,1,0.2,10,learn_rate)
    #agent = PPOAgent(env.datasize, action_size, layer_amount - 1, neurons, epsilon, decay, learn_rate, clip, crit_discount)
    start_time = time.time()

    for e in range(EPISODES):
        episode_correct = 0
        episode_attacks= 0
        episode_correct_attacks=0
        false_positives = 0
        state = env.reset()
        for index in range(batch_size):
            action = agent.act(state)
            next_state, reward, correct, attack, correct_attack, false_pos = env.step(action)
            agent.remember(state, action, reward)
            state = next_state
            episode_correct += correct
            episode_attacks += attack
            episode_correct_attacks += correct_attack
            false_positives += false_pos

        agent.replay(32)
        agent.update_explore_rate(0.5 * (1 + math.cos(e / EPISODES * math.pi)))


        # print(f"batch: {batch}/{batch_amounts} acc: {batch_reward/train_batch}")
        acc = episode_correct / batch_size
        if episode_correct_attacks+false_positives != 0:
            precision = episode_correct_attacks / (episode_correct_attacks + false_positives)

        else:
            precision = 0.0

        if episode_attacks != 0:
            recall = episode_correct_attacks / episode_attacks
        else:
            recall = 0.0
        if precision + recall != 0:
            f1_score = 2 * (precision * recall / (precision + recall))
        else:
            f1_score= 0.0


        print(f"episode: {e}/{EPISODES}, e: {agent.epsilon:.2f}, total acc: {acc}   precision: {precision} recall:{recall} f1:{f1_score} got {episode_attacks}")
        progress[e] = acc
        precision_progress[e] = precision
        recall_progress[e] = recall
        f1_progress[e] = f1_score

        del acc
        del precision
        del recall
        del f1_score
        if e % 50 == 0:
            tf.keras.backend.clear_session()
            gc.collect()




    traintime = time.time() - start_time

    data = pd.read_feather(savefile)
    data.loc[len(data)] = [tau, progress, precision_progress,recall_progress,f1_progress,
                           traintime, EPISODES, 0.0, the_reward, the_penalty, atk_reward, atk_penalty,0.001, env.datasize, "PPO"]

    data.to_feather(savefile)


    #agent.save(f"models/PPO_{layer_amount}x{neurons}_{EPISODES}_{env.datasize}_.h5")

    del agent
    del progress
    del start_time
    del data
    del traintime
    gc.collect()






if __name__ == "__main__":

    run()

