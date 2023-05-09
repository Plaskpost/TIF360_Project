import gymnasium as gym
import torch
import numpy as np
import keyboard


class Controller:

    def __init__(self):
        self.torque = 0.05
        self.action = self.torque*np.ones(8)
        self.joint_keys = [6, 3, 8, 9, 4, 7, 2, 1]
        keyboard.on_press(self.on_press)

    def on_press(self, event):
        if event.name not in str(self.joint_keys):
            return
        index = self.joint_keys.index(int(event.name))
        self.action[index] *= -1


if __name__ == '__main__':

    env = gym.make('Ant-v4', render_mode='human')
    state = env.reset()[0]

    controller = Controller()
    state = torch.tensor(state)

    while True:
        env.render()
        state, reward, done, _, _ = env.step(controller.action)

