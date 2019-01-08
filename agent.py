import random

class Random_Agent:
    def __init__(self):
        self.action_space = ['L', 'R', 'U', 'D', 'N']
        self.opposite_action = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}

    def get_action(self, momentum):
        temp_action_space = self.action_space.copy()
        temp_action_space.remove(self.opposite_action[momentum])
        action = random.choice(temp_action_space)
        return action
