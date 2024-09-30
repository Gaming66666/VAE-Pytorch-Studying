import numpy as np


class ComplexTrafficEnv:
    def __init__(self, num_intersections=4, num_lanes=3, max_steps=100):
        self.num_intersections = num_intersections  # 交叉口数量
        self.num_lanes = num_lanes  # 每个方向的车道数量
        self.max_cars_per_lane = 20  # 每条车道上最多的车辆数量
        self.max_steps = max_steps  # 最大仿真步数
        self.step_count = 0
        self.state = np.zeros((self.num_intersections, 3, 3))  # 状态：每个交叉口每个方向的车流量（左转、直行、右转），以及每个方向的信号灯状态
        self.signal_states = np.zeros((self.num_intersections, 4))  # 信号灯状态：0 红灯，1 绿灯，2 黄灯
        self.action_space = [0, 1, 2]  # 每个交叉口的动作空间：0=红灯，1=绿灯，2=黄灯
        self.lane_flow_rate = np.random.rand(self.num_intersections, 4)  # 每个交叉口的车流量增长率
        self.waiting_time_penalty = -1
        self.throughput_reward = 1

    def reset(self):
        self.state = np.random.randint(0, self.max_cars_per_lane, (self.num_intersections, 3, 3))
        self.signal_states = np.zeros((self.num_intersections, 4))  # 所有信号灯初始化为 红灯
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate([self.state.flatten(), self.signal_states.flatten()])

    def step(self, action):
        reward = 0
        done = False
        for i in range(self.num_intersections):
            if action[i] in self.action_space:
                self.signal_states[i] = action[i]
        for i in range(self.num_intersections):
            for lane in range(3):  # 处理每条车道的左转、直行、右转
                if self.signal_states[i] == 1:  # 绿灯时车辆通行
                    self.state[i, lane] = self._process_lane(self.state[i, lane])
            reward += self._calculate_reward(i)  # 计算每个交叉口的奖励
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        next_state = self._get_observation()
        return next_state, reward, done

    def _process_lane(self, lane_state):
        cars_leaving = int(lane_state * 0.3)
        lane_state = np.maximum(0, lane_state - cars_leaving)
        new_cars = int(np.random.rand() * 5)  # 每步随机有 0-5 辆新车进入
        lane_state = np.minimum(self.max_cars_per_lane, lane_state + new_cars)
        return lane_state

    def _calculate_reward(self, intersection_id):
        waiting_time = np.sum(self.state[intersection_id])  # 等待车辆的数量作为惩罚
        throughput = np.sum(self.signal_states[intersection_id] == 1)  # 绿灯时通行的车辆数
        reward = self.throughput_reward * throughput + self.waiting_time_penalty * waiting_time
        return reward

    def close(self):
        pass


# 测试环境运行
env = ComplexTrafficEnv()
state = env.reset()
done = False
while not done:
    action = [np.random.choice(env.action_space) for _ in range(env.num_intersections)]
    next_state, reward, done = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")

# 深度 Q 网络流程如下：
# 训练代码如下：

import numpy as np
import random
import tensorflow as tf
from collections import deque


# 定义自定义的交通环境
class TrafficEnv:
    def __init__(self, num_intersections=4, max_steps=200):
        self.num_intersections = num_intersections  # 交叉口数量
        self.max_cars_per_lane = 20  # 每条车道的最大车辆数
        self.max_steps = max_steps  # 最大仿真步数
        self.step_count = 0
        self.state = np.zeros((self.num_intersections, 4))  # (交叉口数，方向的车流)
        self.action_space = [0, 1, 2]  # 信号灯状态：0=东西绿灯，1=南北绿灯，2=黄灯
        self.lane_flow_rate = np.random.rand(self.num_intersections, 4)

    def reset(self):
        self.state = np.random.randint(0, self.max_cars_per_lane, (self.num_intersections, 4))
        self.step_count = 0
        return self.state.flatten()

    def step(self, action):
        reward = 0
        done = False
        for i in range(self.num_intersections):
            if action[i] == 0:  # 东西方向绿灯
                self.state[i, 0:2] = np.maximum(0, self.state[i, 0:2] - int(self.state[i, 0:2].sum() * 0.3))
            elif action[i] == 1:  # 南北方向绿灯
                self.state[i, 2:4] = np.maximum(0, self.state[i, 2:4] - int(self.state[i, 2:4].sum() * 0.3))
            elif action[i] == 2:  # 黄灯，减缓车流通行
                self.state[i] = np.maximum(0, self.state[i] - 1)
        self.state += np.random.randint(0, 5, self.state.shape)
        self.state = np.minimum(self.state, self.max_cars_per_lane)  # 避免车道溢出
        reward = -np.sum(self.state)  # 惩罚未通行车辆的数量
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        next_state = self.state.flatten()
        return next_state, reward, done

    def close(self):
        pass


# 定义 DQN 网络
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu '))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.choice([0, 1, 2]) for _ in range(env.num_intersections)]
        act_values = self.model.predict(state)
        return [np.argmax(av) for av in act_values]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            for i in range(len(action)):
                target_f[0][action[i]] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 使用 DQN 与自定义环境
env = TrafficEnv(num_intersections=4, max_steps=200)
state_size = env.num_intersections * 4  # 每个交叉口 4 个方向车流量
action_size = 3  # 每个交叉口 3 个可能的信号灯动作
agent = DQNAgent(state_size, action_size)

episodes = 500
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(200):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
