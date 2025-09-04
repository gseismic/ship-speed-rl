import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.beta2.ship_env import ShipEnv
import numpy as np
import pytest

class TestShipEnvBeta2:
    """测试beta2版本的船舶环境"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.env = ShipEnv(
            dt=2,
            soc_min=0.2,
            soc_max=0.8,
            v_min=0.1,
            v_max=50.0,
            max_time=72,
            eval=False,
            data_file='data/Data_Input.xlsx',
            engine_version='v1',
            reward_type='raw'
        )
    
    def test_env_initialization(self):
        """测试环境初始化"""
        assert self.env.dt == 2
        assert self.env.vmin == 0.1
        assert self.env.vmax == 50.0
        assert self.env.SOC_low == 0.2
        assert self.env.SOC_high == 0.8
        assert self.env.max_time == 72
        assert self.env.engine_version == 'v1'
        assert self.env.reward_type == 'raw'
        assert not self.env.eval
    
    def test_action_space(self):
        """测试动作空间"""
        assert self.env.action_space.shape == ()
        assert self.env.action_space.low == 0.1
        assert self.env.action_space.high == 50.0
        assert self.env.action_space.dtype == np.float64
        
        # 测试动作采样
        samples = np.array([self.env.action_space.sample() for _ in range(100)])
        assert np.all(samples >= 0.1)
        assert np.all(samples <= 50.0)
    
    def test_observation_space(self):
        """测试观察空间"""
        assert 'soc' in self.env.observation_space.spaces
        assert 'r_position' in self.env.observation_space.spaces
        
        # 测试SOC空间
        soc_space = self.env.observation_space.spaces['soc']
        assert soc_space.shape == (1,)
        assert soc_space.low == 0.0
        assert soc_space.high == 1.0
        
        # 测试位置空间
        pos_space = self.env.observation_space.spaces['r_position']
        assert pos_space.low == 0.0
        assert pos_space.high == self.env.max_S_nm
    
    def test_reset(self):
        """测试环境重置"""
        obs, info = self.env.reset()
        
        # 检查观察值
        assert 'soc' in obs
        assert 'r_position' in obs
        assert 0.2 <= obs['soc'][0] <= 0.8  # SOC应在范围内
        assert obs['r_position'][0] == 0.0  # 初始位置应为0
        
        # 检查信息
        assert 'total_fuel' in info
        assert info['total_fuel'] == 0.0
    
    def test_step_basic(self):
        """测试基本步进功能"""
        obs, info = self.env.reset()
        
        # 执行一个动作
        action = 10.0  # 10 m/s的速度
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查返回值
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # 检查SOC变化
        assert 'soc' in next_obs
        assert 'r_position' in next_obs
        
        # 检查位置变化
        assert next_obs['r_position'][0] > obs['r_position'][0]
    
    def test_step_multiple(self):
        """测试多步执行"""
        obs, info = self.env.reset()
        
        for i in range(5):
            action = 15.0  # 15 m/s的速度
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            print(f'步数: {i+1}, 位置: {next_obs["r_position"][0]:.2f}, '
                  f'SOC: {next_obs["soc"][0]:.3f}, 奖励: {reward:.2f}')
            
            if terminated or truncated:
                break
        
        # 检查是否正常执行了多步
        assert i >= 0
    
    def test_engine_version_v2(self):
        """测试v2引擎版本"""
        env_v2 = ShipEnv(engine_version='v2')
        obs, info = env_v2.reset()
        
        # 执行动作
        action = 20.0
        next_obs, reward, terminated, truncated, info = env_v2.step(action)
        
        assert isinstance(reward, (int, float))
        assert 'soc' in next_obs
    
    def test_reward_types(self):
        """测试不同奖励类型"""
        reward_types = ['raw', 'per_nm', 'scaled']
        
        for reward_type in reward_types:
            env = ShipEnv(reward_type=reward_type)
            obs, info = env.reset()
            
            action = 12.0
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(reward, (int, float))
            print(f'奖励类型 {reward_type}: {reward}')
    
    def test_regularization(self):
        """测试正则化功能"""
        env = ShipEnv(regularization_type='Np_Q_Nrpm')
        obs, info = env.reset()
        
        action = 18.0
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (int, float))
    
    def test_eval_mode(self):
        """测试评估模式"""
        env_eval = ShipEnv(eval=True)
        obs, info = env_eval.reset()
        
        action = 25.0
        next_obs, reward, terminated, truncated, info = env_eval.step(action)
        
        assert isinstance(reward, (int, float))
    
    def test_boundary_conditions(self):
        """测试边界条件"""
        obs, info = self.env.reset()
        
        # 测试最小速度
        action = 0.1
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        assert isinstance(reward, (int, float))
        
        # 测试最大速度
        action = 50.0
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        assert isinstance(reward, (int, float))
    
    def test_fuel_consumption(self):
        """测试燃料消耗计算"""
        obs, info = self.env.reset()
        initial_fuel = info['total_fuel']
        
        action = 30.0
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查燃料消耗是否增加
        assert info['total_fuel'] > initial_fuel
        print(f'燃料消耗: {info["total_fuel"]:.4f}')

def test_main():
    """主测试函数"""
    print("开始测试beta2船舶环境...")
    
    # 创建测试实例
    test_env = TestShipEnvBeta2()
    
    # 运行所有测试
    test_env.setup_method()
    
    print("\n=== 测试环境初始化 ===")
    test_env.test_env_initialization()
    
    print("\n=== 测试动作空间 ===")
    test_env.test_action_space()
    
    print("\n=== 测试观察空间 ===")
    test_env.test_observation_space()
    
    print("\n=== 测试环境重置 ===")
    test_env.test_reset()
    
    print("\n=== 测试基本步进 ===")
    test_env.test_step_basic()
    
    print("\n=== 测试多步执行 ===")
    test_env.test_step_multiple()
    
    print("\n=== 测试v2引擎 ===")
    test_env.test_engine_version_v2()
    
    print("\n=== 测试不同奖励类型 ===")
    test_env.test_reward_types()
    
    print("\n=== 测试正则化 ===")
    test_env.test_regularization()
    
    print("\n=== 测试评估模式 ===")
    test_env.test_eval_mode()
    
    print("\n=== 测试边界条件 ===")
    test_env.test_boundary_conditions()
    
    print("\n=== 测试燃料消耗 ===")
    test_env.test_fuel_consumption()
    
    print("\n所有测试完成！")

if __name__ == '__main__':
    test_main()
