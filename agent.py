import random
import numpy as np
from model import *

action_space = ['L', 'R', 'U', 'D', 'N']


class BaseAgent:
    def __init__(self, experience_pool_size, shape):
        self.action_space = action_space
        self.experience_count = 0
        self.experience_pool_size = experience_pool_size
        self.state_shape = shape
        self.state_size = np.prod(shape)
        self.experience_pool = np.zeros(shape=(experience_pool_size, np.prod(shape) * 2 + 3))

    def collect_experience(self, state, action, reward, next_state, done):
        index = self.experience_count % self.experience_pool_size
        action = self.action_space.index(action)
        self.experience_pool[index, :] = np.hstack((state.flatten(), action, reward, next_state.flatten(), done))
        self.experience_count += 1

    def agent_specific_method(self):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, experience_pool_size, shape):
        super().__init__(experience_pool_size=experience_pool_size, shape=shape)

    def get_action(self, state):
        action = random.choice(self.action_space)
        return action


class NNAgent(BaseAgent):
    def __init__(self, shape, epsilon, gamma, learning_rate, mini_batch_size, experience_pool_size
                 , eval_net_threshold, target_net_threshold):
        super().__init__(experience_pool_size=experience_pool_size, shape=shape)
        self.name = 'NN'
        self.eval_net = Net(shape, learning_rate)
        self.target_net = Net(shape, learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.eval_net_threshold = eval_net_threshold
        self.target_net_threshold = target_net_threshold
        self.eval_net_count = 0
        self.target_net_count = 0

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            input_state = np.expand_dims(state, axis=0)
            q_values = self.eval_net.model.predict(input_state)[0]
            action = self.action_space[np.argmax(q_values)]

        return action

    def update_eval_net(self):
        sample_index = random.sample(range(self.experience_pool_size), self.mini_batch_size)
        samples = self.experience_pool[sample_index, :]

        shape = (-1, self.state_shape[0], self.state_shape[1], self.state_shape[2])

        state = samples[:, 0:self.state_size].reshape(shape)
        action = samples[:, self.state_size:self.state_size+1]
        reward = samples[:, self.state_size+1:self.state_size+2]
        next_state = samples[:, self.state_size+2:-1].reshape(shape)
        done = samples[:, -1]

        next_q_values = np.amax(self.target_net.model.predict(next_state), axis=1)
        target = reward.reshape(-1) + self.gamma * next_q_values * (abs(done - 1))

        self.eval_net.model.fit(state, target, epochs=1, verbose=0)

    def update_target_net(self):
        pass

    def agent_specific_method(self):
        # update eval_net_count
        self.eval_net_count += 1

        if self.experience_count >= self.experience_pool_size:
            # update eval net
            if self.eval_net_count > self.eval_net_threshold:
                self.update_eval_net()
                self.eval_net_count = 0
                self.target_net_count += 1

            # update target_net
            if self.target_net_count > self.target_net_threshold:
                self.update_target_net()
                self.target_net_count = 0
