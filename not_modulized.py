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
action_space = [1, 2, 3, 4, 0] #1, 2, 3, 4, 0 represent Left, Right, Up, Down and Nothing respectively
opposite_action = {1:2, 2:1, 3:4, 4:3}
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
    player = [0.7, 0.3, 0.2]
    snake[height//3, width//2:width//2+3] = player
    
    head_y, head_x = np.argwhere(snake == 0.7)[0]
    second_y, second_x = np.argwhere(snake == 0.3)[0]
    tail_y, tail_x = np.argwhere(snake == 0.2)[0]
    
    player_loc = [[head_y, head_x], [second_y, second_x], [tail_y, tail_x]]
    
    # Set food
    food_y, food_x = random.choice(np.argwhere(base+snake == 0))
    base[food_y, food_x] = 0.1

    # reset score
    score = 0
    
    # reset momentum
    momentum = 1
    
    for _ in range(frames):
        # generate new food if no food found
        if np.argwhere(base+snake == 0.1).size == 0:
            food_y, food_x = random.choice(np.argwhere(base+snake == 0))
            base[food_y, food_x] = 0.1
            
        # print image
        snake = arena.copy()
        for idx, val in enumerate(player):
            y, x = player_loc[idx]
            snake[y, x] = val
            
        pool.append(base+snake)
    
        plt.clf()
        plt.imshow(base+snake)
        plt.pause(0.2)
        
        # select an action
        temp_action_space = action_space.copy()
        temp_action_space.remove(opposite_action[momentum])
        action = random.choice(temp_action_space)
        
        # determine new head of snake based on action
        if action == 0:
            action = momentum
            
        if action == 1:
            new_head_y = head_y
            new_head_x = head_x - 1
            
        elif action == 2:
            new_head_y = head_y
            new_head_x = head_x + 1
            
        elif action == 3:
            new_head_y = head_y - 1
            new_head_x = head_x
            
        elif action == 4:
            new_head_y = head_y + 1
            new_head_x = head_x
            
        else:
            print('error')
            break
        
        momentum = action
        
        # determine result of action   
        if [new_head_y, new_head_x] in edges:
            print('You lose! You hit a wall!!!')
            print('Total score:' + str(score))
            break
        
        elif snake[new_head_y, new_head_x] >= 0.2:
            print('You lose! You ate yourself!!!')
            print('Total score:' + str(score))
            break
        
        else:
            player_loc = [[new_head_y, new_head_x]] + player_loc[:-1]
            
            if base[tail_y, tail_x] == 0.1:
                base[tail_y, tail_x] = 0.0
                player[-1] += 0.1
                player.append(0.2)
                player_loc.append([tail_y, tail_x])
                
            if base[new_head_y, new_head_x] == 0.1:
                score += 1
                print('Score!!! +' + str(score) + 'points')
                
            head_y, head_x = new_head_y, new_head_x
            tail_y, tail_x = np.argwhere(snake == 0.2)[0]