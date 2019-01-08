import random
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, height = 20):
        if height < 20:
            print('Height must be larger or equal to 20!!! Initialize the environment again!!!')
        else:
            # Set arena
            self.height = height
            self.width = height // 2
            self.arena = np.zeros(shape=(self.height, self.width), dtype=float)
            self.arena[[0, -1], :] = 0.5
            self.arena[:, [0, -1]] = 0.5

            # Set experience_pool
            self.experience_pool = []

            # Set basic info of each game
            self.base,  self.edges, self.score = [None] * 3

            # Set snake info of each game
            self.snake, self.snake_loc, self.momentum = [None] * 3
            self.head_y, self.head_x, self.second_x, self.second_y, self.tail_x, self.tail_y = [None] * 6

    def render(self):
        plt.ion()
        plt.figure()

    def start_new_game(self):
        # Set base, move and snake
        self.base = self.arena.copy()
        self.snake = self.arena.copy()
        self.edges = np.argwhere(self.base == 0.5).tolist()

        # Set snake
        self.head_y, self.head_x = self.height // 3, self.width // 2
        self.second_y, self.second_x = self.head_y, self.head_x + 1
        self.tail_y, self.tail_x = self.head_y, self.head_x + 2

        self.snake[self.head_y, self.head_x] = 0.7
        self.snake[self.second_y, self.second_x] = 0.6
        self.snake[self.tail_y, self.tail_x] = 0.6
        self.snake_loc = [[self.head_y, self.head_x], [self.second_y, self.second_x], [self.tail_y, self.tail_x]]

        # Set food
        if np.argwhere(self.base + self.snake == 0.3).size == 0:
            food_y, food_x = random.choice(np.argwhere(self.base + self.snake == 0))
            self.base[food_y, food_x] = 0.3

        # Reset score
        self.score = 0

        # Reset momentum
        self.momentum = 'L'
        return self.momentum

    def step(self, action):
        # add the game into experience pool
        self.experience_pool.append([[self.base, self.snake], action])

        # determine new head of snake based on action
        if action == 'N':
            action = self.momentum

        # determine the momentum by the end of step
        self.momentum = action

        # determine the locaiton of new_head if the action is implemented
        if action == 'L':
            new_head_y = self.head_y
            new_head_x = self.head_x - 1

        elif action == 'R':
            new_head_y = self.head_y
            new_head_x = self.head_x + 1

        elif action == 'U':
            new_head_y = self.head_y - 1
            new_head_x = self.head_x

        elif action == 'D':
            new_head_y = self.head_y + 1
            new_head_x = self.head_x

        else:
            print('error')

        '''determine result of action'''
        # lose the game if hitting edges
        if [new_head_y, new_head_x] in self.edges:
            reward = -1
            self.experience_pool[-1].append(reward)
            print('You lose! You hit a wall!!!')
            print('Total score:' + str(self.score))

            is_dead = True
            return self.momentum, is_dead

        # lose the game if eating itself
        elif self.snake[new_head_y, new_head_x] >= 0.6:
            reward = -1
            self.experience_pool[-1].append(reward)
            print('You lose! You ate yourself!!!')
            print('Total score:' + str(self.score))

            is_dead = True
            return self.momentum, is_dead

        else:
            # score if the new_head reaches food
            if self.base[new_head_y, new_head_x] == 0.3:
                reward = 1
                self.experience_pool[-1].append(reward)
                self.score += reward
                print('Score!!! +' + str(reward) + ' point(s)')

            # update the snake_loc and snake board based on new move
            self.snake[new_head_y, new_head_x] = 0.7
            self.snake[self.head_y, self.head_x] = 0.6
            self.snake_loc = [[new_head_y, new_head_x]] + self.snake_loc

            if self.base[self.tail_y, self.tail_x] == 0.3:
                self.base[self.tail_y, self.tail_x] = 0.0

            else:
                self.snake[self.tail_y, self.tail_x] = 0.0
                self.snake_loc.pop(-1)

                # update current head and current tail
                self.head_y, self.head_x = self.snake_loc[0]
                self.tail_y, self.tail_x = self.snake_loc[-1]

            # generate new food if no food found
            if np.argwhere(self.base + self.snake == 0.3).size == 0:
                food_y, food_x = random.choice(np.argwhere(self.base + self.snake == 0))
                self.base[food_y, food_x] = 0.3

            plt.clf()
            plt.imshow(self.base + self.snake)
            plt.pause(0.1)

            is_dead = False
            return self.momentum, is_dead



