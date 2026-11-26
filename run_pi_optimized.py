import argparse
import os
import sys
from fallDownDetectYolo_optimized import FallDownDetectYoloOptimized

if __name__ == "__main__":
    # 获取当前脚本所在目录的上级目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    default_yolov5_path = os.path.join(project_root, "yolov5")
    default_weight_path = os.path.join(project_root, "modelweight", "model.onnx")
    
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description="树莓派优化版跌倒检测器 - 专门针对树莓派性能优化",
        epilog="优化特性: 降低分辨率、跳帧处理、CPU优化、内存管理",
    )

    # 添加参数
    parser.add_argument("--yolov5_path", type=str, default=default_yolov5_path, help="YOLOv5 本地路径")
    parser.add_argument("--weight_path", type=str, default=default_weight_path, help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--image_path", type=str, help="待检测的图片路径")
    parser.add_argument("--video_path", type=str, help="待检测的视频路径")
    parser.add_argument("--model_type", type=str, default='onnx', choices=['pt', 'onnx'], help="模型类型：pt 或 onnx")
    parser.add_argument("--input_type", type=str, default='video', choices=['image', 'video'], help="输入类型：image 或 video")
    parser.add_argument("--save", action='store_true', help="是否保存检测结果")
    parser.add_argument("--save_path", type=str, default=None, help="检测结果保存路径")
    parser.add_argument("--fps", type=int, default=5, help="目标帧率（建议5-10fps）")
    parser.add_argument("--skip_frames", type=int, default=3, help="跳帧数量（每N+1帧处理1帧）")
    parser.add_argument("--input_size", type=int, default=320, help="模型输入尺寸（建议320或416）")

    # 解析参数
    args = parser.parse_args()

    # 参数验证
    if args.input_type == "image" and not args.image_path:
        print("错误: 在图像模式下，必须指定 --image_path")
        sys.exit(1)
    if args.input_type == "video" and not args.video_path:
        print("未指定 video_path，将使用摄像头捕获视频流")
        args.video_path = None
    
    # 性能建议
    print("=== 树莓派优化建议 ===")
    print(f"目标帧率: {args.fps} FPS")
    print(f"跳帧设置: 每{args.skip_frames + 1}帧处理1帧")
    print(f"输入尺寸: {args.input_size}x{args.input_size}")
    print(f"模型类型: {args.model_type}")
    print("=====================")
        
    # 创建优化版检测器
    detector = FallDownDetectYoloOptimized(
        yolov5_path=args.yolov5_path,
        weight_path=args.weight_path,
        image_path=args.image_path if args.input_type == 'image' else '',
        video_path=args.video_path if args.input_type == 'video' else None,
        pt_or_onnx=args.model_type
    )

    # 更新优化参数
    detector.input_size = args.input_size
    detector.skip_frames = args.skip_frames

    # 运行检测器
    if args.input_type == "image":
        detector.img_inference(save=args.save, save_path=args.save_path)
    elif args.input_type == "video":
        detector.video_inference_optimized(
            save=args.save,
            save_path=args.save_path if args.save_path else "output_optimized.mp4",
            fps=args.fps
        ) 