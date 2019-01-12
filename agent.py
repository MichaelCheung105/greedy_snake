import random
#from model import Net

action_space = ['L', 'R', 'U', 'D', 'N']
opposite_action = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}

class Base_Agent:
    def __init__(self):
        self.action_space = action_space
        self.opposite_action = opposite_action

    def get_action(self, momentum):
        temp_action_space = self.action_space.copy()
        temp_action_space.remove(self.opposite_action[momentum])
        action = random.choice(temp_action_space)
        return action

class Random_Agent(Base_Agent):
    pass

class NN_Agent(Base_Agent):
    pass