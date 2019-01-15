import random
import numpy as np
from model import Net

action_space = ['L', 'R', 'U', 'D', 'N']


class BaseAgent:
    def __init__(self, experience_pool_size):
        self.action_space = action_space
        self.experience_count = 0
        self.experience_pool_size = experience_pool_size
        self.experience_pool = [None] * experience_pool_size

    def collect_experience(self, state, action, reward, next_state, done):
        index = self.experience_count % self.experience_pool_size
        self.experience_pool[index] = (state, action, reward, next_state, done)
        self.experience_count += 1

    def agent_specific_method(self):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, experience_pool_size):
        super().__init__(experience_pool_size=experience_pool_size)

    def get_action(self, state):
        action = random.choice(self.action_space)
        return action


class NNAgent(BaseAgent):
    def __init__(self, shape, epsilon, gamma, learning_rate, mini_batch_size, experience_pool_size
                 , eval_net_threshold, target_net_threshold):
        super().__init__(experience_pool_size=experience_pool_size)
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
        samples = random.sample(self.experience_pool, self.mini_batch_size)

        for sample in samples:
            state, action, reward, next_state, done = sample

            if done:
                target = reward
            else:
                input_state = np.expand_dims(next_state, axis=0)
                next_q_values = self.target_net.model.predict(input_state)[0]
                target = reward + self.gamma * np.max(next_q_values)

            input_state = np.expand_dims(state, axis=0)
            self.eval_net.model.fit(input_state, target, epochs=1, verbose=0)

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
