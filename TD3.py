import tensorflow as tf
import numpy as np
import random
from collections import deque

class TD3Agent:

    def __init__(self, state_size, action_size, layer_amount, neurons, buffer_size, gamma, tau, policy_noise, noise_clip, policy_freq, learn_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.learn_rate = learn_rate
        self.memory = deque(maxlen=self.buffer_size)
        self.actor = self._build_actor(layer_amount, neurons)
        self.critic1 = self._build_critic(layer_amount, neurons)
        self.critic2 = self._build_critic(layer_amount, neurons)
        self.target_actor = self._build_actor(layer_amount, neurons)
        self.target_critic1 = self._build_critic(layer_amount, neurons)
        self.target_critic2 = self._build_critic(layer_amount, neurons)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(self.learn_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(self.learn_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(self.learn_rate)

    def _build_actor(self, layer_amount, neurons):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_size))
        for _ in range(layer_amount):
            model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="tanh"))
        return model

    def _build_critic(self, layer_amount, neurons):
        state_input = tf.keras.layers.Input(shape=self.state_size)
        action_input = tf.keras.layers.Input(shape=self.action_size)
        state_h1 = state_input
        action_h1 = action_input
        for _ in range(layer_amount):
            state_h1 = tf.keras.layers.Dense(neurons, activation="relu")(state_h1)
            action_h1 = tf.keras.layers.Dense(neurons, activation="relu")(action_h1)
        concat = tf.keras.layers.Concatenate()([state_h1, action_h1])
        concat_h1 = tf.keras.layers.Dense(neurons, activation="relu")(concat)
        output = tf.keras.layers.Dense(1)(concat_h1)
        return tf.keras.Model(inputs=[state_input, action_input], outputs=output)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def _soft_update(self, target, source):
        target_weights = target.get_weights()
        source_weights = source.get_weights()

        new_weights = []
        for i, target_weight in enumerate(target_weights):
            source_weight = source_weights[i]
            new_weight = self.tau * source_weight + (1 - self.tau) * target_weight
            new_weight = new_weight.reshape(target_weight.shape)
            new_weights.append(new_weight)

        target.set_weights(new_weights)

    def replay(self, batch_size):


        minibatch = random.sample(self.memory, batch_size)

        states = np.array([m[0] for m in minibatch])
        states = np.squeeze(states)
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])

        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        # Train critic networks
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(states)
            target_actions = tf.clip_by_value(target_actions, -1, 1)
            target_critic1_value = self.target_critic1([states, target_actions])
            target_critic2_value = self.target_critic2([states, target_actions])

            # Compute target Q value
            target_q_values = tf.minimum(target_critic1_value, target_critic2_value)
            target_q_values = rewards[:, None] + self.gamma * target_q_values

            # Compute critic loss
            critic1_value = self.critic1([states, actions])
            critic2_value = self.critic2([states, actions])
            critic1_loss = tf.math.reduce_mean(tf.math.square(target_q_values - critic1_value))
            critic2_loss = tf.math.reduce_mean(tf.math.square(target_q_values - critic2_value))

        critic1_grads = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # Delayed policy updates
        if self.policy_freq % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic1([states, new_actions]))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Soft update target networks
            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

    def act(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.actor(state)[0]
        noise = np.random.normal(0, self.policy_noise, size=self.action_size)
        action = tf.clip_by_value(action + noise, -1, 1)
        return action

    def update_explore_rate(self, epsilon):
        self.policy_noise = epsilon

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.actor.save(f"models/td3models/a_{name}")
        self.critic1.save(f"models/td3models/c1_{name}")
        self.critic2.save(f"models/td3models/c2_{name}")