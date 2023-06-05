import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import random

class PPOAgent:
    def __init__(self, state_size, action_size, layer_amount, neurons, epsilon, decay, learn_rate,
                 clip_ratio=0.2, critic_discount=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = decay
        self.learning_rate = learn_rate
        self.clip_ratio = clip_ratio
        self.critic_discount = critic_discount
        self.actor = self._build_actor(layer_amount, neurons)
        self.critic = self._build_critic(layer_amount, neurons)

    def _build_actor(self, layer_amount, neurons):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(neurons, activation="relu")(inputs)
        for i in range(layer_amount):
            x = layers.Dense(neurons, activation="relu")(x)
        outputs = layers.Dense(self.action_size, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self._actor_loss, optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _actor_loss(self, y_true, y_pred):
        advantage = tf.stop_gradient(self.critic(y_true)) - self.critic_discount
        old_pred = tf.stop_gradient(y_pred)
        prob = tf.reduce_sum(y_true * y_pred, axis=-1)
        old_prob = tf.reduce_sum(y_true * old_pred, axis=-1)
        r_theta = prob / old_prob
        clipped = tf.clip_by_value(r_theta, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        entropy = tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=-1)
        return -tf.reduce_mean(tf.minimum(clipped, r_theta * advantage)) - 0.01 * entropy

    def _build_critic(self, layer_amount, neurons):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(neurons, activation="relu")(inputs)
        for i in range(layer_amount):
            x = layers.Dense(neurons, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def act(self, state):
        tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        prob = self.actor(tensor)[0]
        action = np.random.choice(self.action_size, p=prob.numpy())
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([transition[0] for transition in minibatch])
        states = np.squeeze(states)
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])

        with tf.GradientTape(persistent=True) as tape:
            values = self.critic(np.array(states))

            # Compute advantages
            advantages = rewards - tf.squeeze(values)

            # Normalize advantages
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

            # Compute old probabilities for clipping
            old_probs = self.actor(states)

            # Convert actions to one-hot encoding
            action_masks = tf.one_hot(actions, self.action_size)

            # Compute probabilities of actions taken
            probs = tf.reduce_sum(action_masks * old_probs, axis=1)

            # Compute ratio of new and old probabilities
            ratio = probs / tf.stop_gradient(tf.reduce_sum(action_masks * old_probs, axis=1))

            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Compute actor loss
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Compute critic loss
            critic_loss = tf.reduce_mean(tf.square(rewards - values))

            # Compute total loss
            total_loss = actor_loss + critic_loss

        # Compute gradients
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(total_loss, self.critic.trainable_variables)

        # Apply gradients
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def update_explore_rate(self, epsilon):
        self.epsilon = epsilon


    def load(self, name):
        self.actor = tf.keras.models.load_model(f"models/{name}", custom_objects={'_actor_loss': self._actor_loss})
        self.critic = tf.keras.models.load_model(f"models/PPO_Critic_model/critic_{name}")

    def save(self, name):
        self.actor.save(f"models/{name}")
        self.critic.save(f"models/PPO_Critic_model/critic_{name}")