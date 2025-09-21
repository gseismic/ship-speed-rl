import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym 
from collections import deque 
import random 
import matplotlib.pyplot as plt 
import copy 
import pandas as pd 
import time 

# TODO: 确保所有的是张量才可能反向传播 
import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
from src.environment.gamma2.ship_env import ShipEnv 

class ShipOptimizer(nn.Module): 
    """神经网络模型，用于优化船舶航行""" 
    def __init__(self, input_dim, output_dim, hidden_dim=128): 
        super(ShipOptimizer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 输出层激活函数
        if output_dim == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Tanh()
    
    def forward(self, state):
        # 状态归一化
        norm_state = self.normalize_state(state)
        raw_output = self.net(norm_state)
        return self.output_activation(raw_output)
    
    def normalize_state(self, state):
        """归一化状态输入"""
        # 提取状态值
        soc = state["soc"]
        r_position = state["r_position"]
        r_time = state["r_time"]
        V_wind = state["V_wind"]
        alpha_wind = state["alpha_wind"]
        
        # 归一化
        norm_soc = soc
        norm_r_position = r_position / self.max_position
        norm_r_time = r_time / self.max_time
        norm_V_wind = V_wind / 20.0
        norm_alpha_wind = alpha_wind / 360.0
        
        return torch.tensor([norm_soc, norm_r_position, norm_r_time, 
                            norm_V_wind, norm_alpha_wind], dtype=torch.float32)

class ShipOptimizationAgent:
    """船舶优化智能体 - 完整修正版"""
    def __init__(self, env, learning_rate=0.001, gamma=0.99, batch_size=64):
        self.env = env
        self.state_dim = 5  # soc, r_position, r_time, V_wind, alpha_wind
        self.action_dim = 1 if env.engine_version in ['v1', 'v2'] else 2
        
        # 获取环境的最大值
        self.max_position = env.max_S_nm
        self.max_time = env.max_time
        
        # 神经网络模型
        self.model = ShipOptimizer(self.state_dim, self.action_dim)
        self.model.max_position = self.max_position
        self.model.max_time = self.max_time
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练参数
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        
        # 损失权重
        self.fuel_weight = 1.0
        self.soc_weight = 0.5
        self.constraint_weight = 0.3
        self.goal_weight = 2.0
        
        # 训练统计
        self.total_steps = 0
    
    def state_to_tensor(self, state_dict):
        """将状态字典转换为张量"""
        return {
            'soc': state_dict['soc'][0],
            'r_position': state_dict['r_position'][0],
            'r_time': state_dict['r_time'][0],
            'V_wind': state_dict['V_wind'][0],
            'alpha_wind': state_dict['alpha_wind'][0]
        }
    
    def get_action(self, state_dict, deterministic=False):
        """获取动作 - 修正版"""
        state = self.state_to_tensor(state_dict)
        
        if deterministic:
            with torch.no_grad():
                action_norm = self.model(state)
        else:
            # 在训练模式下，确保计算梯度
            action_norm = self.model(state)
        
        # 反归一化到动作空间
        if self.action_dim == 1:
            v_ship = action_norm.item() * (self.env.vmax - self.env.vmin) + self.env.vmin
            return np.array([v_ship])
        else:
            v_ship = action_norm[0].item() * (self.env.vmax - self.env.vmin) + self.env.vmin
            q_ratio = (action_norm[1].item() + 1) / 2  # 从[-1,1]映射到[0,1]
            return np.array([v_ship, q_ratio])
    
    def remember(self, state_dict, action, reward, next_state_dict, done):
        """存储经验"""
        self.memory.append((state_dict, action, reward, next_state_dict, done))
    
    def compute_loss(self, batch):
        """计算损失函数 - 修正版"""
        states, actions, rewards, next_states, dones = batch
        
        # 初始化损失张量
        fuel_loss = torch.tensor(0.0, requires_grad=True)
        soc_penalty = torch.tensor(0.0, requires_grad=True)
        constraint_penalty = torch.tensor(0.0, requires_grad=True)
        goal_reward = torch.tensor(0.0, requires_grad=True)
        
        # 处理批次中的每个样本
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            
            # 1. 燃油消耗损失
            fuel_loss += torch.tensor(-reward, requires_grad=True)
            
            # 2. SOC边界惩罚
            soc = state['soc'][0]
            if soc < self.env.SOC_low:
                penalty = (self.env.SOC_low - soc) ** 2
            elif soc > self.env.SOC_high:
                penalty = (soc - self.env.SOC_high) ** 2
            else:
                penalty = 0.0
            soc_penalty += torch.tensor(penalty, requires_grad=True)
            
            # 3. 设备约束惩罚（从环境中获取）
            # 模拟环境执行一步（不实际改变环境状态）
            env_copy = copy.deepcopy(self.env)
            env_copy.state = state
            _, _, _, _, info = env_copy.step(action)
            
            # 检查约束违反
            penalty = 0.0
            if info['Esignal1'] != 0 or info['Esignal2'] != 0:
                penalty += 1.0
            if info['Battery_state'] != 0:
                penalty += 0.5
            constraint_penalty += torch.tensor(penalty, requires_grad=True)
            
            # 4. 目标奖励（最大化）
            if done and reward > 0:  # 成功到达终点
                goal_reward += torch.tensor(reward, requires_grad=True)
        
        # 平均损失
        num_samples = len(states)
        fuel_loss /= num_samples
        soc_penalty /= num_samples
        constraint_penalty /= num_samples
        goal_reward /= num_samples
        
        # 组合损失
        loss = (
            self.fuel_weight * fuel_loss +
            self.soc_weight * soc_penalty +
            self.constraint_weight * constraint_penalty -
            self.goal_weight * goal_reward
        )
        
        return loss
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 计算损失
        loss = self.compute_loss((states, actions, rewards, next_states, dones))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=1000):
        """训练循环 - 修正版"""
        rewards = []
        losses = []
        successes = []
        fuel_consumptions = []
        start_time = time.time()
        
        for episode in range(episodes):
            state_dict, _ = self.env.reset()
            episode_reward = 0
            episode_fuel = 0
            done = False
            step_count = 0
            
            while not done:
                # 获取动作
                action = self.get_action(state_dict)
                
                # 执行动作
                next_state_dict, reward, done, _, info = self.env.step(action)
                
                # 存储经验
                self.remember(state_dict, action, reward, next_state_dict, done)
                
                # 更新状态
                state_dict = next_state_dict
                episode_reward += reward
                episode_fuel += info["delta_fuel"]
                step_count += 1
                self.total_steps += 1
                
                # 经验回放
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        losses.append(loss)
            
            # 记录结果
            rewards.append(episode_reward)
            fuel_consumptions.append(episode_fuel)
            successes.append(1 if info.get('succ', False) else 0)
            
            # 每10轮打印一次进度
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / (episode + 1) if episode > 0 else 0
                remaining_time = avg_time_per_episode * (episodes - episode)
                
                success_rate = np.mean(successes[-10:]) * 100 if len(successes) >= 10 else 0
                avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else 0
                avg_fuel = np.mean(fuel_consumptions[-10:]) if len(fuel_consumptions) >= 10 else 0
                
                print(f"Episode {episode}/{episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Fuel: {episode_fuel:.2f} kg | "
                      f"Avg Fuel: {avg_fuel:.2f} kg | "
                      f"Success: {success_rate:.1f}% | "
                      f"Steps: {step_count} | "
                      f"Remaining: {remaining_time/60:.1f} min")
        
        # 保存模型
        torch.save(self.model.state_dict(), "ship_optimizer.pth")
        
        # 绘制训练结果
        self.plot_training(rewards, losses, successes, fuel_consumptions)
    
    def plot_training(self, rewards, losses, successes, fuel_consumptions):
        """绘制训练结果 - 修正版"""
        plt.figure(figsize=(18, 12))
        
        # 奖励曲线
        plt.subplot(221)
        plt.plot(rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        # 损失曲线
        plt.subplot(222)
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        
        # 成功率曲线
        plt.subplot(223)
        success_rate = np.convolve(successes, np.ones(50)/50, mode='valid') if len(successes) >= 50 else successes
        plt.plot(success_rate)
        plt.title("Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.ylim([0, 100])
        
        # 燃油消耗曲线
        plt.subplot(224)
        plt.plot(fuel_consumptions)
        plt.title("Fuel Consumption")
        plt.xlabel("Episode")
        plt.ylabel("Fuel (kg)")
        
        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.show()
    
    def test(self, num_episodes=10):
        """测试训练好的模型 - 修正版"""
        total_fuel = 0
        success_count = 0
        episode_details = []
        
        for episode in range(num_episodes):
            state_dict, _ = self.env.reset()
            done = False
            episode_fuel = 0
            positions = []
            times = []
            socs = []
            actions = []
            
            while not done:
                action = self.get_action(state_dict, deterministic=True)
                next_state_dict, _, done, _, info = self.env.step(action)
                state_dict = next_state_dict
                episode_fuel += info['delta_fuel']
                
                # 记录轨迹
                positions.append(info['position'])
                times.append(info['time'])
                socs.append(info['soc'])
                actions.append(action)
            
            total_fuel += episode_fuel
            if info.get('succ', False):
                success_count += 1
            
            episode_details.append({
                'fuel': episode_fuel,
                'success': info.get('succ', False),
                'positions': positions,
                'times': times,
                'socs': socs,
                'actions': actions
            })
            
            print(f"Test Episode {episode+1} | Fuel: {episode_fuel:.2f} kg | "
                  f"Success: {info.get('succ', False)}")
        
        # 打印汇总结果
        print(f"\nAverage Fuel Consumption: {total_fuel/num_episodes:.2f} kg")
        print(f"Success Rate: {success_count/num_episodes*100:.1f}%")
        
        # 绘制最优轨迹
        if episode_details:
            best_episode = min(episode_details, key=lambda x: x['fuel'])
            self.plot_optimal_trajectory(best_episode)
    
    def plot_optimal_trajectory(self, episode):
        """绘制最优轨迹"""
        plt.figure(figsize=(15, 10))
        
        # 位置-时间图
        plt.subplot(221)
        plt.plot(episode['times'], episode['positions'])
        plt.title("Position vs Time")
        plt.xlabel("Time (hours)")
        plt.ylabel("Position (nm)")
        
        # SOC变化图
        plt.subplot(222)
        plt.plot(episode['times'], episode['socs'])
        plt.title("SOC vs Time")
        plt.xlabel("Time (hours)")
        plt.ylabel("SOC")
        plt.axhline(y=self.env.SOC_low, color='r', linestyle='--', label='SOC Min')
        plt.axhline(y=self.env.SOC_high, color='g', linestyle='--', label='SOC Max')
        plt.legend()
        
        # 速度变化图
        plt.subplot(223)
        speeds = [a[0] for a in episode['actions']]
        plt.plot(episode['times'][:-1], speeds)
        plt.title("Speed vs Time")
        plt.xlabel("Time (hours)")
        plt.ylabel("Speed (m/s)")
        plt.axhline(y=self.env.vmin, color='r', linestyle='--', label='Min Speed')
        plt.axhline(y=self.env.vmax, color='g', linestyle='--', label='Max Speed')
        plt.legend()
        
        # Q_m_ratio变化图（如果适用）
        if self.action_dim == 2:
            plt.subplot(224)
            q_ratios = [a[1] for a in episode['actions']]
            plt.plot(episode['times'][:-1], q_ratios)
            plt.title("Q_m Ratio vs Time")
            plt.xlabel("Time (hours)")
            plt.ylabel("Q_m Ratio")
            plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig("optimal_trajectory.png")
        plt.show()

# 主程序
if __name__ == "__main__":
    # 创建环境（eval模式）
    env = ShipEnv(
        eval=True,
        engine_version='v3',  # 使用最复杂的引擎版本
        reward_type='raw',
        regularization_type='SOC',
        data_file='data/Data_Input.xlsx'
    )
    
    # 创建并训练智能体
    agent = ShipOptimizationAgent(env)
    agent.train(episodes=500)
    
    # 测试训练好的模型
    agent.test(num_episodes=10)

    # 关闭环境
    env.close()