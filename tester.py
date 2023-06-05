import random
import concurrent.futures
import pandas as pd
from memory_profiler import profile
from RL_IDS_env import MyEnv
#from RL_IDS_env_dynamic import MyEnv
from DQN import DQNAgent
from DDQN import DDQNAgent
import gc
import numpy as np
import time
import os
from PPO2 import PPOAgent
import multiprocessing as mp
import tensorflow as tf
from statistics import mode


def run(file_path):

    savefile = ("results/PPO_common_attacks_results.feather")
    # savefile = ("results/DQN10K_transfer_learning_results.feather")
    # df = pd.read_feather("dataset/2019/Dos_DNS_bin_69feat_CIC2019.feather")
    #df = pd.read_feather(f"dataset/singleattack/{file_path}")
    parts = file_path.split('_')
    df = pd.read_feather("dataset/test_common_classes_69feat_CIC2017.feather")
    env = MyEnv(df)
    del df
    gc.collect()
    action_size = 15
    layer_amount = 3
    neurons_amount = 16
    EPISODES = 50
    batch_size = 3200
    # PPO Agent
    Pagent = PPOAgent(env.features, action_size, layer_amount - 1, neurons_amount, 0, 0, 0.001, 0.99, 0.95)
    Pagent.load("PPO_common_classes_3x16_15754_69_.h5")

    # DQN Agents
    #Dagent5 = DQNAgent(env.features, action_size, layer_amount -1, neurons_amount, 0, 0, 0.001)
    #Dagent10 = DQNAgent(env.features, action_size, layer_amount - 1, neurons_amount, 0, 0, 0.001)

    # Dagent5.load("modified_random_classifier_DQN_3x16_15754_69_.h5")
    # Dagent5.load("classifier_DQN_3x16_15754_69_.h5")
    # Dagent10.load("DQN_3x16_10000_69_.h5")
    # Dagent10.load("sampled_nondos_9_14_classifier_DQN_3x16_15754_69_.h5")

    # DDQN Agents
    # DDagent5 = DDQNAgent(env.features, action_size, layer_amount -1, neurons_amount, 0, 0, 0.001)
    # DDagent10 = DDQNAgent(env.features, action_size, layer_amount - 1, neurons_amount, 0, 0, 0.001)
    # DDagent20 = DDQNAgent(env.features, action_size, layer_amount - 1, neurons_amount, 0, 0, 0.001)
    #DDagent5 = DDQNAgent(env.features, action_size, layer_amount - 1, neurons_amount, 0, 0, 0.001)
    #DDagent5.load("DDQN_class_mod2_3x16_15754_69_.h5")

    # DDagent5.load("DDQN_3x16_5000_69_.h5")
    # DDagent10.load("DDQN_3x16_10000_69_.h5")
    # DDagent20.load("DDQN_3x16_20000_69_.h5")

    progress = np.zeros(EPISODES)
    precision_progress = np.zeros(EPISODES)
    recall_progress = np.zeros(EPISODES)
    f1_progress = np.zeros(EPISODES)
    total_atk= 0
    total_benign = 0
    total_correct_attack = 0
    total_correct_benign = 0
    attack_accuracy = {}
    for i in range(0, 15):
        class_name = i
        attack_accuracy[class_name] = {"success_count": 0, "appearance_count": 0}

    state = env.reset()
    start_time = time.time()
    for e in range(EPISODES):
        episode_correct = 0
        episode_attacks = 0
        episode_correct_attacks = 0
        false_positives = 0

        for val in range(batch_size):
            #action1 = Dagent5.act(state)
            #action2 = Dagent10.act(state)
            # action = DDagent5.act(state)
            # action = DDagent10.act(state)
            # action1 = DDagent20.act(state)
            action = Pagent.act(state)


            # nondosatk = [9,10,11,12,13,14]
            #
            # if action1 != 0:
            #     action = action1
            # elif action2 in nondosatk:
            #     action = action2
            # else:
            #     action = 0
            # action = mode([action1,action2,action3,action4,action5])

            next_state, reward, correct, attack, correct_attack, false_pos, label = env.step(action)
            state = next_state
            episode_correct += correct
            episode_attacks += attack


            episode_correct_attacks += correct_attack
            false_positives = false_pos
            attack_accuracy[label]["appearance_count"] += 1
            attack_accuracy[label]["success_count"] += correct

        acc = episode_correct / batch_size
        if episode_correct_attacks + false_positives != 0:
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
            f1_score = 0.0

        # print(f"acc: {acc}\n",
        #       f"precision: {precision}\n",
        #       f"recall: {recall}\n",
        #       f"f1: {f1_score}\n",
        #       )

        progress[e] = acc
        precision_progress[e] = precision
        recall_progress[e] = recall
        f1_progress[e] = f1_score

    traintime = time.time() - start_time

    # feather dont like int as keys so change them to str before saving to file
    attack_accuracy = {str(key): value for key, value in attack_accuracy.items()}
    atk_penalty = {str(key): value for key, value in env.attack_penalty.items()}
    data = pd.read_feather(savefile)
    data.loc[len(data)] = ["PPO common classes", progress, precision_progress, recall_progress, f1_progress,
                           traintime, EPISODES, atk_penalty, attack_accuracy,
                            total_atk, total_benign, total_correct_attack, total_correct_benign]



    data.to_feather(savefile)
    # {parts[0]}


if __name__ == "__main__":
    # folder_path = "dataset/singleattack"
    # files = os.listdir(folder_path)
    # files = files[1:]
    # for file_path in files:
    #     file_name = os.path.basename(file_path)
    #     run(file_name)

    run("asd")
