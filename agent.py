import random
from model import Net

action_space = ['L', 'R', 'U', 'D', 'N']

class Base_Agent:
    def __init__(self):
        self.action_space = action_space
        self.experience_pool = []
        self.brain = None

    def get_action(self):
        action = random.choice(self.action_space)
        return action

    def collect_experience(self, state, action, reward, next_state):
        self.experience_pool.append([state, action, reward, next_state])


class Random_Agent(Base_Agent):
    def __init__(self):
        super().__init__()


class NN_Agent(Base_Agent):
    def __init__(self):
        super().__init__()
        self.brain = Net()
