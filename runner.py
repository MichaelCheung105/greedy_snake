from environment import Environment
from agent import *

# Config Parameters
episode = 1000
frames = 10**6
height = 20
frame_rate = 0.01
agent = 'NN'  # random / NN
NN_layers = 3
epsilon = 0.5

# Parameters based on config
width = height // 2

if __name__ == "__main__":
    env = Environment(height=height, width=width, frame_rate=frame_rate)

    if agent == 'random':
        agent = Random_Agent()
    elif agent == 'NN':
        agent = NN_Agent(shape=(height, width, NN_layers), epsilon=epsilon)
    else:
        print('No such agent!!!')
        exit()

    for game in range(episode):
        state = env.start_new_game()

        for frame in range(frames):
            # determin state, action and reward
            action = agent.get_action(state)
            next_state, reward, is_dead, info = env.step(action)
            agent.collect_experience(state, action, reward, next_state)
            state = next_state

            # update eval net and target net
            agent.update_eval_net()
            agent.update_target_net()

            if info is not None:
                print(info)

            if is_dead:
                break

