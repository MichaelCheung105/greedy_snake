from environment import Environment
from agent import *

# Config Parameters
episode = 1000
frames = 10**6
height = 20
frame_rate = 0.01
agent = 'random'  # random / NN
layers = 3
epsilon = 0
gamma = 0.9
learning_rate = 0.05
mini_batch_size = 30
experience_pool_size = 100
eval_net_threshold = 30
target_net_threshold = 10


# Parameters based on config
width = height // 2

if __name__ == "__main__":
    # Set the environment
    env = Environment(height=height, width=width, frame_rate=frame_rate)

    # Determine which agent to use
    if agent == 'random':
        agent = Random_Agent(experience_pool_size=experience_pool_size)

    elif agent == 'NN':
        agent = NN_Agent(shape=(height, width, layers)
                         , epsilon=epsilon
                         , gamma=gamma
                         , learning_rate=learning_rate
                         , mini_batch_size=mini_batch_size
                         , experience_pool_size=experience_pool_size
                         , eval_net_threshold=eval_net_threshold
                         , target_net_threshold=target_net_threshold
                         )

    else:
        print('No such agent!!!')
        exit()

    # Start training
    for game in range(episode):
        state = env.start_new_game()

        for frame in range(frames):
            # determine state, action and reward
            action = agent.get_action(state)
            next_state, reward, is_dead, info = env.step(action)
            agent.collect_experience(state, action, reward, next_state)
            state = next_state

            # execute specific method for the agent
            agent.agent_specific_method()

            if info is not None:
                print(info)

            if is_dead:
                break

