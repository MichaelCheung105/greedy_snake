import random
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, height=20, frame_rate=0.1):
        if height < 20:
            print('Height must be larger or equal to 20!!! Initialize the environment again!!!')
        else:
            # Set frame_rate
            self.frame_rate = frame_rate

            # Set arena
            self.height = height
            self.width = height // 2
            self.arena = np.zeros(shape=(self.height, self.width), dtype=float)

            # Set experience_pool
            self.experience_pool = []

            # Set the starting base, edges and score of each game
            self.base,  self.edges, self.score = [None] * 3

            # Set snake info of each game
            self.snake, self.snake_loc, self.momentum = [None] * 3
            self.head_y, self.head_x, self.second_x, self.second_y, self.tail_x, self.tail_y = [None] * 6

            # Set food info of each game
            self.food = None

            # Set image
            plt.ion()
            plt.figure()

    def start_new_game(self):
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

        # Set food
        self.food = self.arena.copy()

        if np.argwhere(self.base + self.food + self.snake == 0.3).size == 0:
            food_y, food_x = random.choice(np.argwhere(self.base + self.food + self.snake == 0))
            self.food[food_y, food_x] = 0.3

        # Reset score
        self.score = 0

        # Reset momentum
        self.momentum = 'L'

        # Show initial stage
        plt.clf()
        plt.imshow(self.base + self.food + self.snake)
        plt.pause(self.frame_rate)

        return self.momentum

    def step(self, action):
        # add the game into experience pool
        state = np.stack([self.base, self.food, self.snake])
        self.experience_pool.append([state, action])

        # determine new head of snake based on action
        if action == 'N':
            action = self.momentum

        # determine the momentum
        self.momentum = action

        # determine the location of new_head if the action is implemented
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
        # update the snake_loc and snake board based on new move
        self.snake[new_head_y, new_head_x] = 0.7
        self.snake[self.head_y, self.head_x] = 0.6
        self.snake[self.tail_y, self.tail_x] = 0.0
        self.snake_loc = [[new_head_y, new_head_x]] + self.snake_loc

        # lose the game if hitting edges
        if [new_head_y, new_head_x] in self.edges:
            is_dead = True
            reward = -1
            print('You lose! You hit a wall!!!')
            print('Total score:' + str(self.score))

        # lose the game if eating itself
        elif [new_head_y, new_head_x] in self.snake_loc[1:]:
            is_dead = True
            reward = -1
            print('You lose! You ate yourself!!!')
            print('Total score:' + str(self.score))

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
                print('Score!!! +' + str(reward) + ' point(s)')

                # generate new food
                food_y, food_x = random.choice(np.argwhere(self.base + self.food + self.snake == 0))
                self.base[food_y, food_x] = 0.3

            else:
                reward = 0

            plt.clf()
            plt.imshow(self.base + self.food + self.snake)
            plt.pause(self.frame_rate)

        next_state = np.stack([self.base, self.food, self.snake])
        self.experience_pool[-1].append(reward)
        self.experience_pool[-1].append(next_state)

        return self.momentum, is_dead
