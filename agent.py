import random
from model import Net

action_space = ['L', 'R', 'U', 'D', 'N']

class Base_Agent:
    def __init__(self):
        self.action_space = action_space
        self.experience_pool = []
        self.brain = None

    def collect_experience(self, state, action, reward, next_state):
        self.experience_pool.append([state, action, reward, next_state])


class Random_Agent(Base_Agent):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        action = random.choice(self.action_space)
        return action


class NN_Agent(Base_Agent):
    def __init__(self, shape, epsilon):
        super().__init__()
        self.brain = Net(shape, epsilon)

    def get_action(self, state):
        action = self.brain.suggest(state, self.action_space)
        return action

    def update_eval_net(self):
        pass

    def update_target_net(self):
        pass

