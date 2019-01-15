import random
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, height, width, frame_rate):
        if height < 20:
            print('Height must be larger or equal to 20!!! Initialize the environment again!!!')
        else:
            # Set frame_rate
            self.frame_rate = frame_rate

            # Set arena
            self.height = height
            self.width = width
            self.arena = np.zeros(shape=(self.height, self.width), dtype=float)

            # Set the starting base, edges and score of each game
            self.base,  self.edges, self.score = [None] * 3

            # Set snake info of each game
            self.snake, self.snake_loc, self.momentum = [None] * 3
            self.head_y, self.head_x, self.second_x, self.second_y, self.tail_x, self.tail_y = [None] * 6
            self.opposite_action = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}

            # Set food info of each game
            self.food = None

            # Set image
            plt.ion()
            plt.figure()

    def reset(self):
        # Set the base and define edges
        self.base = self.arena.copy()
        self.base[[0, -1], :] = 1.0
        self.base[:, [0, -1]] = 1.0

        self.edges = np.argwhere(self.base == 1.0).tolist()

        # Set snake
        self.snake = self.arena.copy()
        self.head_y, self.head_x = self.height // 3, self.width // 2
        self.second_y, self.second_x = self.head_y, self.head_x + 1
        self.tail_y, self.tail_x = self.head_y, self.head_x + 2

        self.snake[self.head_y, self.head_x] = 0.7
        self.snake[self.second_y, self.second_x] = 0.6
        self.snake[self.tail_y, self.tail_x] = 0.6
        self.snake_loc = [[self.head_y, self.head_x], [self.second_y, self.second_x], [self.tail_y, self.tail_x]]

        self.momentum = 'L'

        # Set food
        self.food = self.arena.copy()

        if np.argwhere(self.base + self.food + self.snake == 0.3).size == 0:
            food_y, food_x = random.choice(np.argwhere(self.base + self.food + self.snake == 0))
            self.food[food_y, food_x] = 0.3

        # Reset score
        self.score = 0

        # Show initial stage
        plt.clf()
        plt.imshow(self.base + self.food + self.snake)
        plt.pause(self.frame_rate)

        state = np.stack([self.base, self.food, self.snake], axis=-1)
        return state

    def step(self, action):
        # determine new head of snake based on action
        if action == 'N' or self.opposite_action[action] == self.momentum:
            processed_action = self.momentum

        else:
            processed_action = action

        # update the momentum
        self.momentum = processed_action

        # determine the location of new_head if the action is implemented
        if processed_action == 'L':
            new_head_y = self.head_y
            new_head_x = self.head_x - 1

        elif processed_action == 'R':
            new_head_y = self.head_y
            new_head_x = self.head_x + 1

        elif processed_action == 'U':
            new_head_y = self.head_y - 1
            new_head_x = self.head_x

        elif processed_action == 'D':
            new_head_y = self.head_y + 1
            new_head_x = self.head_x

        else:
            return print('error: incorrect action input')
            exit()

        '''determine result of action'''
        # update the snake_loc and snake board based on new move
        self.snake[new_head_y, new_head_x] = 0.7
        self.snake[self.head_y, self.head_x] = 0.6
        self.snake[self.tail_y, self.tail_x] = 0.0
        self.snake_loc = [[new_head_y, new_head_x]] + self.snake_loc

        # lose the game if hitting edges
        if [new_head_y, new_head_x] in self.edges:
            is_dead = True
            reward = -1
            info = 'You lose! You hit a wall! Total score: ' + str(self.score)

        # lose the game if eating itself
        elif [new_head_y, new_head_x] in self.snake_loc[1:]:
            is_dead = True
            reward = -1
            info = 'You lose! You ate yourself! Total score: ' + str(self.score)

        else:
            is_dead = False

            # if the food is at the tail of the snake, transform the food into the new tail of the snake
            if self.food[self.tail_y, self.tail_x] == 0.3:
                self.food[self.tail_y, self.tail_x] = 0.0
                self.snake[self.tail_y, self.tail_x] = 0.6
            else:
                self.snake_loc.pop(-1)

            # update current head and current tail
            self.head_y, self.head_x = self.snake_loc[0]
            self.tail_y, self.tail_x = self.snake_loc[-1]

            # score if the new_head reaches food
            if self.food[new_head_y, new_head_x] == 0.3:
                reward = 1
                self.score += reward
                info = 'Score!!! +' + str(reward) + ' point(s)'

                # generate new food
                food_y, food_x = random.choice(np.argwhere(self.base + self.food + self.snake == 0))
                self.food[food_y, food_x] = 0.3

            else:
                reward = 0
                info = None

            plt.clf()
            plt.imshow(self.base + self.food + self.snake)
            plt.pause(self.frame_rate)

        next_state = np.stack([self.base, self.food, self.snake], axis=-1)
        return next_state, reward, is_dead, info
