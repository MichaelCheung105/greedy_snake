import random
from model import Net

action_space = ['L', 'R', 'U', 'D', 'N']

class Base_Agent:
    def __init__(self):
        self.action_space = action_space
        self.experience_pool = []
        self.name = None
        self.brain = None

    def agent_specific_method(self):
        pass

    def collect_experience(self, state, action, reward, next_state):
        self.experience_pool.append([state, action, reward, next_state])


class Random_Agent(Base_Agent):
    def __init__(self):
        super().__init__()
        self.name = 'Random_Agent'

    def get_action(self, state):
        action = random.choice(self.action_space)
        return action


class NN_Agent(Base_Agent):
    def __init__(self, shape, epsilon, eval_net_threshold, target_net_threshold):
        super().__init__()
        self.name = 'NN'
        self.brain = Net(shape, epsilon)
        self.eval_net_count = 0
        self.target_net_count = 0
        self.eval_net_threshold = eval_net_threshold
        self.target_net_threshold = target_net_threshold

    def get_action(self, state):
        action = self.brain.suggest(state, self.action_space)
        return action

    def update_eval_net(self):
        pass

    def update_target_net(self):
        pass

    def agent_specific_method(self):
        # update eval_net_count
        self.eval_net_count += 1

        # update eval net
        if self.eval_net_count > self.eval_net_threshold:
            self.update_eval_net()
            self.eval_net_count = 0
            self.target_net_count += 1

        # update target_net
        if self.target_net_count > self.target_net_threshold:
            self.update_target_net()
            self.target_net_count = 0
