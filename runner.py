from environment import Environment
from agent import *

# Config Parameters
episode = 10**5
frames = 10**1000
height = 20
frame_rate = 0.01
agent = 'NN'  # random / NN
layers = 3
epsilon = 0.1
gamma = 1
learning_rate = 0.05
mini_batch_size = 500
experience_pool_size = 1000
eval_net_threshold = 30
target_net_threshold = 10


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
            agent.agent_specific_method()

            if info is not None:
                print(info)

            if done:
                break

        if game % 1000 == 0:
            agent.eval_net.model.save_weights("eval_net_model.h5")
            agent.target_net.model.save_weights("target_net_model.h5")

