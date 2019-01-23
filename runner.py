from environment import Environment
from agent import *

# Config Parameters
episode = 10**5
model_saving_threshold = 1000
frames = 10**1000
height = 20
frame_rate = 0.01
agent = 'random'  # random / NN
layers = 3
epsilon = 1
gamma = 1
learning_rate = 0.05
mini_batch_size = 30
experience_pool_size = 100
eval_net_threshold = 30
target_net_threshold = 10
is_enable_ddqn = True


# Parameters based on config
width = height // 2

if __name__ == "__main__":
    # Set the environment
    env = Environment(height=height, width=width, frame_rate=frame_rate)

    # Determine which agent to use
    if agent == 'random':
        agent = RandomAgent(experience_pool_size=experience_pool_size, shape=(height, width, layers))

    elif agent == 'NN':
        agent = NNAgent(shape=(height, width, layers)
                        , epsilon=epsilon
                        , gamma=gamma
                        , learning_rate=learning_rate
                        , mini_batch_size=mini_batch_size
                        , experience_pool_size=experience_pool_size
                        , eval_net_threshold=eval_net_threshold
                        , target_net_threshold=target_net_threshold
                        , is_enable_ddqn=is_enable_ddqn
                        )

    else:
        print('No such agent!!!')
        exit()

    # Start training
    for game in range(episode):
        print('episode:', game+1)
        state = env.reset()

        for frame in range(frames):
            # determine state, action and reward
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.collect_experience(state, action, reward, next_state, done)
            state = next_state

            # execute specific method for the agent
            agent.agent_specific_method(game, model_saving_threshold)

            if info is not None:
                print(info)

            if done:
                break

