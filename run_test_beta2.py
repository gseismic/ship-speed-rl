#!/usr/bin/env python3
"""
快速测试beta2船舶环境的脚本
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def quick_test():
    """快速测试函数"""
    try:
        print("正在导入beta2船舶环境...")
        from src.environment.beta2.ship_env import ShipEnv
        print("✓ 导入成功")
        
        print("\n正在创建环境...")
        env = ShipEnv(
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
        print("✓ 环境创建成功")
        
        print("\n正在重置环境...")
        obs, info = env.reset()
        print(f"✓ 环境重置成功")
        print(f"  初始SOC: {obs['soc'][0]:.3f}")
        print(f"  初始位置: {obs['r_position'][0]:.2f} 海里")
        print(f"  初始燃料: {info['total_fuel']:.4f}")
        
        print("\n正在测试基本步进...")
        action = 15.0  # 15 m/s
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ 步进成功")
        print(f"  新位置: {next_obs['r_position'][0]:.2f} 海里")
        print(f"  新SOC: {next_obs['soc'][0]:.3f}")
        print(f"  奖励: {reward:.2f}")
        print(f"  燃料消耗: {info['total_fuel']:.4f}")
        
        print("\n正在测试多步执行...")
        for i in range(3):
            action = 20.0  # 20 m/s
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(f"  步数 {i+1}: 位置={next_obs['r_position'][0]:.2f}, "
                  f"SOC={next_obs['soc'][0]:.3f}, 奖励={reward:.2f}")
            
            if terminated or truncated:
                print(f"  环境在第{i+1}步终止")
                break
        
        print("\n✓ 所有测试完成！环境运行正常。")
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请检查文件路径和依赖是否正确安装")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__': 
    print("=" * 50) 
    print("beta2船舶环境快速测试")  
    print("=" * 50) 
    
    quick_test()
