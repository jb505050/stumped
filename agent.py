import numpy as np
from numpy import random
from task import Task

class findAgent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size = (self.state_size, self.action_size),
            scale = 10)


        #self.w = np.random.normal(
        #    size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
        #    scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        self.actCount = 0
        self.actTrack = []
        self.rewardTrack = []

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        #this portion of the code might be useless
        # Learn, if at end of episode
        if done:
            self.learn()
            self.rewardTrack.append(self.total_reward)
            self.actTrack.append(self.actCount)
            self.actCount =0

    def act(self, sim_pose, target_pos, previous_act, r1, r2):
        realWorld = sim_pose[2].astype('int') #define the real world z position of the copter
        goalWorld = target_pos[2].astype('int') #define the goal z position of the copter
        speed = previous_act[1].astype('int') #obtain rotor speed from previous step

        #adjust rotor speed based on change of reward function across the steps
        #if copter is above where it needs to be slow rotors
        if realWorld > goalWorld:
            rotorSpeed = speed * (1 - (abs(r1 - r2)))
        #if copter is below where it needs to be speed rotors up
        else:
            rotorSpeed = speed * (1 + (abs(r1 - r2)))
        return action = [rotorSpeed, rotorSpeed, rotorSpeed, rotorSpeed]


    #this entire function might be useless now given how I have architected the act function
    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(10 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        