# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:00:28 2019

@author: CHEUNMI2
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# Set game size
height = 20
width = height//2
games = 1000
frames = 100000

# Prompt error
if height < 20:
    print('Height must be larger than 20!!!')

# Set arena
arena = np.zeros(shape=(height, width), dtype=float)
arena[[0,-1], :] = 0.5
arena[:, [0, -1]] = 0.5

# Set initial condition of game
action_space = ['L', 'R', 'U', 'D', 'N']
opposite_action = {'L':'R', 'R':'L', 'U':'D', 'D':'U'}
pool = []
plt.ion()
plt.figure()

# Loop the game
for game in range(games):
    # Set base, move and snake
    base = arena.copy()
    snake = arena.copy()
    edges = np.argwhere(base == 0.5).tolist()
    
    # Set snake  
    head_y, head_x = height//3, width//2
    second_y, second_x = head_y, head_x + 1
    tail_y, tail_x = head_y, head_x + 2
    
    snake[head_y, head_x] = 0.7
    snake[second_y, second_x] = 0.6
    snake[tail_y, tail_x] = 0.6
    snake_loc = [[head_y, head_x], [second_y, second_x], [tail_y, tail_x]]       
    
    # reset score
    score = 0
    
    # reset momentum
    momentum = 'L'
    
    for _ in range(frames):
        # generate new food if no food found
        if np.argwhere(base+snake == 0.3).size == 0:
            food_y, food_x = random.choice(np.argwhere(base+snake == 0))
            base[food_y, food_x] = 0.3
            
        pool.append([base, snake])
    
        plt.clf()
        plt.imshow(base+snake)
        plt.pause(0.1)
        
        # select an action
        temp_action_space = action_space.copy()
        temp_action_space.remove(opposite_action[momentum])
        action = random.choice(temp_action_space)
        
        # determine new head of snake based on action
        if action == 'N':
            action = momentum
            
        momentum = action
            
        if action == 'L':
            new_head_y = head_y
            new_head_x = head_x - 1
            
        elif action == 'R':
            new_head_y = head_y
            new_head_x = head_x + 1
            
        elif action == 'U':
            new_head_y = head_y - 1
            new_head_x = head_x
            
        elif action == 'D':
            new_head_y = head_y + 1
            new_head_x = head_x
            
        else:
            print('error')
            break
        
        # determine result of action
        # lose the game if hitting edges
        if [new_head_y, new_head_x] in edges:
            print('You lose! You hit a wall!!!')
            print('Total score:' + str(score))
            break
        
        # lose the game if eating itself
        elif snake[new_head_y, new_head_x] >= 0.6:
            print('You lose! You ate yourself!!!')
            print('Total score:' + str(score))
            break
        
        else:
            # score if the new_head reaches food
            if base[new_head_y, new_head_x] == 0.3:
                score += 1
                print('Score!!! +' + str(score) + ' point')
            
            # update the snake_loc and snake board based on new move
            snake[new_head_y, new_head_x] = 0.7
            snake[head_y, head_x] = 0.6
            snake_loc = [[new_head_y, new_head_x]] + snake_loc
            
            if base[tail_y, tail_x] == 0.3:
                base[tail_y, tail_x] = 0.0
                
            else:
                snake[tail_y, tail_x] = 0.0
                snake_loc.pop(-1)
                
            # update current head and current tail
            head_y, head_x = snake_loc[0]
            tail_y, tail_x = snake_loc[-1]