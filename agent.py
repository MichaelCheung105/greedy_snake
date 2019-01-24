import random
import numpy as np
from model import *

action_space = ['L', 'R', 'U', 'D']


class BaseAgent:
    def __init__(self, experience_pool_size, shape):
        self.name = None
        self.is_test_mode = False
        self.action_space = action_space
        self.experience_count = 0
        self.experience_pool_size = experience_pool_size
        self.state_shape = shape
        self.state_size = np.prod(shape)
        self.experience_pool = np.zeros(shape=(experience_pool_size, np.prod(shape) * 2 + 3))

    def get_action(self, state):
        action = random.choice(self.action_space)
        return action

    def collect_experience(self, state, action, reward, next_state, done):
        index = self.experience_count % self.experience_pool_size
        action = self.action_space.index(action)
        self.experience_pool[index, :] = np.hstack((state.flatten(), action, reward, next_state.flatten(), done))
        self.experience_count += 1

    def agent_specific_method(self, *args, **kwargs):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, experience_pool_size, shape):
        super().__init__(experience_pool_size=experience_pool_size, shape=shape)
        self.name = 'random'


class NNAgent(BaseAgent):
    def __init__(self, shape, epsilon, gamma, learning_rate, mini_batch_size, experience_pool_size
                 , eval_net_threshold, target_net_threshold, model_saving_threshold, is_enable_ddqn):
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
        self.eval_net_update_count = 0
        self.target_net_update_count = 0
        self.model_saving_threshold = model_saving_threshold
        self.is_enable_ddqn = is_enable_ddqn

    def get_action(self, state):
        if self.is_test_mode or np.random.rand() >= self.epsilon:
            input_state = np.expand_dims(state, axis=0)
            q_values = self.eval_net.model.predict(input_state)[0]
            action = self.action_space[np.argmax(q_values)]
        else:
            action = random.choice(self.action_space)

        return action

    def update_eval_net(self):
        sample_index = random.sample(range(self.experience_pool_size), self.mini_batch_size)
        samples = self.experience_pool[sample_index, :]

        shape = (-1, self.state_shape[0], self.state_shape[1], self.state_shape[2])

        state = samples[:, 0:self.state_size].reshape(shape)
        action = samples[:, self.state_size:self.state_size+1].reshape(-1).astype(int)
        reward = samples[:, self.state_size+1:self.state_size+2].reshape(-1)
        next_state = samples[:, self.state_size+2:-1].reshape(shape)
        done = samples[:, -1]

        if self.is_enable_ddqn:
            max_q_index = np.argmax(self.eval_net.model.predict(next_state), axis=1)
            next_q_values = self.target_net.model.predict(next_state)[(range(self.mini_batch_size), max_q_index)]
        else:
            next_q_values = np.amax(self.target_net.model.predict(next_state), axis=1)

        q_values = self.eval_net.model.predict(state)
        target = reward + self.gamma * next_q_values * (abs(done - 1))
        q_values[(range(self.mini_batch_size), action)] = target

        self.eval_net.model.train_on_batch(state, q_values)

    def update_target_net(self):
        self.target_net.model.set_weights(self.eval_net.model.get_weights())

    def agent_specific_method(self, game):
        if not self.is_test_mode:
            # save model every 'model_saving_threshold' games
            if game % self.model_saving_threshold == 0:
                self.eval_net.model.save_weights("eval_net_model.h5")
                self.target_net.model.save_weights("target_net_model.h5")

            # update eval_net_count
            self.eval_net_count += 1

            if self.experience_count >= self.experience_pool_size:
                # update eval net
                if self.eval_net_count > self.eval_net_threshold:
                    self.update_eval_net()
                    self.eval_net_count = 0
                    self.eval_net_update_count += 1
                    self.target_net_count += 1

                # update target_net
                if self.target_net_count > self.target_net_threshold:
                    self.update_target_net()
                    self.target_net_count = 0
                    self.target_net_update_count += 1
