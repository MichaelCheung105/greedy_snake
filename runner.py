from environment import Environment
from agent import Random_Agent

episode = 1000
frames = 10**6
height = 20
frame_rate = 0.05

if __name__ == "__main__":
    env = Environment(height=height, frame_rate=0.05)
    agent = Random_Agent()

    for game in range(episode):
        momentum = env.start_new_game()

        for frame in range(frames):
            action = agent.get_action(momentum)
            momentum, is_dead = env.step(action)

            if is_dead:
                break