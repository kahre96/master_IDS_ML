import pyarrow.feather as feather
import pandas as pd
import numpy as np
from RL_IDS_env import MyEnv
#from RL_IDS_env_sys import MyEnv
# from RL_IDS_env_dynamic_reward import MyEnv
# from RL_IDS_env_dynamic import MyEnv
from DQN import DQNAgent
from DDQN import DDQNAgent
from PPO2 import PPOAgent
from TD3 import TD3Agent
import time
import random
import tensorflow as tf
import gc
import math

random.seed(42)


def run():
    action_size = 2

    # change df
    # change save
    # change data
    # savefile = "results/classrewards_2017_DQN_results.feather"
    savefile = "results/DQN_discount_results.feather"
    df = pd.read_feather("dataset/norm_bin_69feat_CIC2017.feather")
    #df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    env = MyEnv(df)
    del df
    gc.collect()

    batch_size = 320
    epsilon = 0.999
    penalty = -1 # -10
    reward = 1  # 1
    learn_rate = 0.001
    layer_amount = 3
    neurons_amount = 16
    EPISODES = 2000
    # EPISODES = 1
    clip = 0.99  # 0.99     0.2 0.4 0.6 0.8 0.9 0.99
    crit_discount = 0.95  # 0.5   0.95td3??
    decay = 0.996
    #atk_reward = 1  # 1 / 2
    # atk_penalty = {1: -120,
    #                2: -90,
    #                3: -100,
    #                4: -170,
    #                5: -190,
    #                6: -250,
    #                7: -250,
    #                8: -250,
    #                9: -250,
    #                10: -75,
    #                11: -75,
    #                12: -75,
    #                13: -75,
    #                14: -75
    #                }
    tau = 0.01  #
    noise_clip = 0.7  #
    noise = 0.9
    cf = 0.00001
    #env.set_reward(reward, penalty)
    #env.set_atk_reward(atk_reward, atk_penalty)
    # e_array = [0.25,0.5,1,2,3]
    # change_array= [0.0,0.001,0.0001,0.00001]

    discount_array=[0.001, 0.01, 0.1, 0.3, 0.5, 0.7,0.9,0.99]

    atk_reward = 1
    atk_penalty= -30

    for discount in discount_array:
        env.set_reward(reward, penalty)
        env.set_atk_reward(atk_reward, atk_penalty)
        train_model(layer_amount, neurons_amount, EPISODES, batch_size, epsilon, decay, env, action_size,
                    learn_rate, savefile, reward, penalty, atk_reward, atk_penalty, clip, crit_discount, tau, noise_clip,
                    cf, noise, discount)


def train_model(layer_amount, neurons, EPISODES, batch_size, epsilon, decay, env, action_size,
                learn_rate, savefile, the_reward, the_penalty, atk_reward, atk_penalty, clip,
                crit_discount, tau, n_clip, cf, noise, discount):
    progress = np.zeros(EPISODES)
    precision_progress = np.zeros(EPISODES)
    recall_progress = np.zeros(EPISODES)
    f1_progress = np.zeros(EPISODES)

    # tracking succes of each attack individualy
    attack_accuracy = {}
    for i in range(0, 15):
        class_name = i
        attack_accuracy[class_name] = {"success_count": 0, "appearance_count": 0}

    # different RL implementations change DQN to DDQN to use DDQN
    #agent = TD3Agent(env.features, action_size, layer_amount - 1, neurons, 2000, 0.99, tau, noise, n_clip, 10, learn_rate)
    #agent = PPOAgent(env.features, action_size, layer_amount - 1, neurons, epsilon, decay, learn_rate, clip, crit_discount)
    agent = DQNAgent(env.features, action_size, layer_amount - 1, neurons, epsilon, decay, learn_rate, discount)
    # agent.load("DQN_3x16_10000_69_.h5")
    start_time = time.time()
    total_atk= 0
    total_benign = 0
    total_correct_attack = 0
    total_correct_benign = 0


    for e in range(EPISODES):
        # variables to track accuracy and f1
        episode_correct = 0
        episode_attacks = 0
        episode_correct_attacks = 0
        false_positives = 0
        state = env.reset()
        for index in range(batch_size):
            action = agent.act(state)
            #next_state, reward, correct, attack, correct_attack, false_pos, label = env.step(np.argmax(action))
            next_state, reward, correct, attack, correct_attack, false_pos, label = env.step(action)
            agent.remember(state, action, reward)
            state = next_state
            episode_correct += correct
            episode_attacks += attack
            episode_correct_attacks += correct_attack
            false_positives += false_pos

            # grabbing accuracy of each attacktype durign the end of training to get an idea of training accuracy
            if e > EPISODES - 100:
                attack_accuracy[label]["appearance_count"] += 1
                attack_accuracy[label]["success_count"] += correct
                if attack == 1:
                    total_atk += 1
                    if correct:
                        total_correct_attack += 1
                else:
                    total_benign += 1
                    if correct:
                        total_correct_benign += 1

        # trainign the model using minibatch
        agent.replay(32)
        # changes the exploration rate according to a cos curve with a lot of exploration early
        agent.update_explore_rate(0.5 * (1 + math.cos(e / EPISODES * math.pi)))

        # calculating accurcy and f1
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

        print(f"episode: {e}/{EPISODES},"
              #f" e: {agent.epsilon:.2f},"
              f" total acc: {acc:.4f} "
              f"  precision: {precision:.4f} "
              f"recall:{recall:.4f}"
              f" f1:{f1_score:.4f}"
              f" got {episode_attacks}"
              # f" Bening: {env.penalty[0]}"
              # f" attack: {env.penalty[1]}"
              )
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

    for attacktype, values in attack_accuracy.items():
        if values['appearance_count'] == 0:
            print(f"{attacktype}: no apprearances")
        else:
            print(f"{attacktype}: {(values['success_count'] / values['appearance_count'])}")

    traintime = time.time() - start_time

    # feather dont like int as keys so change them to str before saving to file
    #attack_accuracy = {str(key): value for key, value in attack_accuracy.items()}
    #atk_penalty = {str(key): value for key, value in atk_penalty.items()}
    # save the data to a result file
    #agent.save(f"PPO_common_classes_{layer_amount}x{neurons}_{EPISODES}_{env.features}_.h5")
    data = pd.read_feather(savefile)
    data.loc[len(data)] = [discount, progress, precision_progress, recall_progress, f1_progress,
                           traintime, EPISODES,
                            total_atk, total_benign, total_correct_attack, total_correct_benign]

    data.to_feather(savefile)

    # save the model


    del agent
    del progress
    del start_time
    del data
    del traintime
    gc.collect()


if __name__ == "__main__":
    run()
