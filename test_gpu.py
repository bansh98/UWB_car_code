#!/usr/bin/env python3
"""
GPU测试脚本
用于验证GPU模式和autocast修复是否正常工作
"""

import torch
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def test_gpu():
    """测试GPU是否可用"""
    print("🔍 GPU测试开始...")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        
        # 测试GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU总内存: {gpu_memory:.1f} GB")
        
        # 测试autocast
        try:
            with torch.amp.autocast('cuda', enabled=True):
                x = torch.randn(1, 3, 640, 640).cuda()
                print("✅ autocast测试通过")
        except Exception as e:
            print(f"❌ autocast测试失败: {e}")
            return False
            
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

def test_yolo_model():
    """测试YOLO模型加载"""
    print("\n🔍 YOLO模型测试开始...")
    
    try:
        from fallDownDetectYolo import FallDownDetectYolo
        
        # 设置路径
        yolov5_path = os.path.join(project_root, "yolov5")
        weight_path = os.path.join(project_root, "modelweight", "yolov5_best.pt")
        
        print(f"YOLOv5路径: {yolov5_path}")
        print(f"权重路径: {weight_path}")
        
        # 检查文件是否存在
        if not os.path.exists(weight_path):
            print(f"❌ 权重文件不存在: {weight_path}")
            return False
        
        # 创建检测器
        detector = FallDownDetectYolo(
            yolov5_path=yolov5_path,
            weight_path=weight_path,
            image_path='',
            pt_or_onnx='pt',
            video_path=None
        )
        
        print("✅ YOLO模型加载成功")
        
        # 测试推理
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 进行推理
        with torch.no_grad():
            results = detector.model(test_image)
        
        print("✅ YOLO推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ YOLO模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始GPU和YOLO模型测试\n")
    
    # 测试GPU
    gpu_ok = test_gpu()
    
    if gpu_ok:
        # 测试YOLO模型
        yolo_ok = test_yolo_model()
        
        if yolo_ok:
            print("\n🎉 所有测试通过！GPU模式可以正常使用。")
            print("\n现在可以运行:")
            print("python simple_tracking.py")
        else:
            print("\n❌ YOLO模型测试失败")
    else:
        print("\n❌ GPU测试失败")

if __name__ == "__main__":
    main() 