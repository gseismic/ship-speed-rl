from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

class EpisodeRewardCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []  # 记录每个episode的总reward
        self.episode_costs = []  # 记录每个episode的总reward
        self.current_episode_rewards = None  # 因为n_envs=4,所以需要4个累积器
        
    def _on_step(self) -> bool:
        # 将当前步的奖励加到当前episode的累积奖励中
        # print(self.locals['rewards'])
        # print(list(self.locals.keys()))
        # print(self.locals['infos'])
        # print(self.locals['dones'])
        # print(self.locals['infos']['lateral_cost_money'])
        # raise
        if self.current_episode_rewards is None:
            self.current_episode_rewards = self.locals['rewards']
        else:
            self.current_episode_rewards += self.locals['rewards']
        
        # 检查哪些环境完成了一个episode
        # print(self.locals)
        dones = self.locals['dones']  # 获取结束标志
        for i, done in enumerate(dones):
            if done:
                # 记录完成的episode的总reward
                self.episode_rewards.append(self.current_episode_rewards[i])
                # self.episode_costs.append(self.locals['infos'][i]['total_cost_money'])
                # print(self.episode_rewards)
                # 重置该环境的累积奖励
                self.current_episode_rewards[i] = 0
                
        return True
    
    def get_rewards(self):
        """返回所有完成的episode的奖励"""
        return self.episode_rewards
    
    def get_total_costs(self):
        return self.episode_costs


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # 每个时间步记录奖励
        # 注意：self.locals['rewards']是一个列表，包含每个环境在当前时间步的奖励
        self.rewards.append(np.mean(self.locals['rewards']))
        return True

    def get_rewards(self):
        """返回所有完成的episode的奖励"""
        return self.rewards

    

class SaveBestModelCallback(BaseCallback):
    def __init__(self, eval_env,
                 check_freq: int, 
                 n_eval_episodes: int = 3, 
                 model_save_path: str = "best_model", 
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.model_save_path = model_save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 在评估环境中运行当前策略
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,       # 每次评估的 episode 数
                deterministic=True,
            )
            # 如果平均奖励超过历史最佳，则保存模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.model_save_path)
                if self.verbose > 0:
                    print(f"New best model with mean reward: {mean_reward}")
        return True
