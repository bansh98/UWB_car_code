#!/usr/bin/env python3
"""
摄像头测试脚本
用于测试不同摄像头配置的稳定性
"""

import cv2
import time
import sys

def test_camera_config(camera_index, api_name, api_value):
    """测试特定的摄像头配置"""
    print(f"\n🔍 测试摄像头: 索引={camera_index}, API={api_name}")
    
    try:
        cap = cv2.VideoCapture(camera_index, api_value)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头")
            return False
        
        # 设置参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 等待初始化
        time.sleep(2)
        
        # 测试读取
        success_count = 0
        total_frames = 10
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
                print(f"  帧 {i+1}: ✅ 成功")
            else:
                print(f"  帧 {i+1}: ❌ 失败")
            time.sleep(0.1)
        
        success_rate = success_count / total_frames * 100
        print(f"📊 成功率: {success_rate:.1f}% ({success_count}/{total_frames})")
        
        if success_rate >= 80:
            print(f"✅ 摄像头配置可用")
            cap.release()
            return True
        else:
            print(f"❌ 摄像头配置不稳定")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始摄像头测试\n")
    
    # 测试不同的配置
    configs = [
        (0, 'DirectShow', cv2.CAP_DSHOW),
        (0, 'Auto', cv2.CAP_ANY),
        (0, 'MSMF', cv2.CAP_MSMF),
        (1, 'DirectShow', cv2.CAP_DSHOW),
        (2, 'DirectShow', cv2.CAP_DSHOW),
    ]
    
    working_configs = []
    
    for camera_index, api_name, api_value in configs:
        if test_camera_config(camera_index, api_name, api_value):
            working_configs.append((camera_index, api_name))
    
    print(f"\n📋 测试结果:")
    if working_configs:
        print("✅ 可用的摄像头配置:")
        for i, (index, api) in enumerate(working_configs, 1):
            print(f"  {i}. 索引={index}, API={api}")
        
        print(f"\n💡 推荐使用:")
        best_config = working_configs[0]
        print(f"python simple_tracking.py --camera_index {best_config[0]} --camera_api {best_config[1].lower()}")
    else:
        print("❌ 没有找到可用的摄像头配置")
        print("💡 建议:")
        print("   1. 检查摄像头连接")
        print("   2. 关闭其他使用摄像头的程序")
        print("   3. 重启摄像头或电脑")

if __name__ == "__main__":
    main() 