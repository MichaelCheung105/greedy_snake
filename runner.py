from environment import Environment
from agent import *

episode = 1000
frames = 10**6
height = 20
frame_rate = 0.01
agent = 'random'  # random / NN

if __name__ == "__main__":
    env = Environment(height=height, frame_rate=frame_rate)

    if agent == 'random':
        agent = Random_Agent()
    elif agent == 'NN':
        agent = NN_Agent()
    else:
        print('No such agent!!!')
        exit()

    for game in range(episode):
        state = env.start_new_game()

        for frame in range(frames):
            action = agent.get_action()
            next_state, reward, is_dead, info = env.step(action)
            agent.collect_experience(state, action, reward, next_state)
            state = next_state

            if info is not None:
                print(info)

            if is_dead:
                break

