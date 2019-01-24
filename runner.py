from environment import Environment
from agent import *

# Config Parameters
episode = 10**10
frames = 10**1000
height = 7
width = 7
frame_rate = 0.01
agent = 'NN'  # random / NN
layers = 3
epsilon = 0.5
gamma = 1
learning_rate = 0.05
mini_batch_size = 100
experience_pool_size = 500
eval_net_threshold = 50
target_net_threshold = 10
model_saving_threshold = 200
observe_rl_threshold = 100
enter_test_mode_threshold = 100
revert_to_train_threshold = 20
is_enable_ddqn = False


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
                        , model_saving_threshold=model_saving_threshold
                        )

    else:
        print('No such agent!!!')
        exit()


    # set train_counter and test_counter
    train_counter = 0
    test_counter = 0

    # Start training
    for game in range(episode):
        print('episode:', game+1)
        state = env.reset()

        if not agent.is_test_mode:
            train_counter += 1
        else:
            test_counter += 1

        if train_counter == enter_test_mode_threshold:
            agent.is_test_mode = True
            train_counter = 0

        if test_counter == revert_to_train_threshold:
            agent.is_test_mode = False
            test_counter = 0

        for frame in range(frames):
            # determine state, action and reward
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.collect_experience(state, action, reward, next_state, done)
            state = next_state

            # execute specific method for the agent
            agent.agent_specific_method(game)

            if info is not None:
                print(info)
                print('is test mode:', agent.is_test_mode)
                print('# of times eval_net was updated:', agent.eval_net_update_count)
                print('# of times target_net was updated:', agent.target_net_update_count)

            if done:
                break

