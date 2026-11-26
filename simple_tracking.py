import argparse
import os
import sys
import time
import serial
import torch
from fallDownDetectYolo import FallDownDetectYolo

class SimpleTracking:
    """
    简化的目标跟踪器，只保留核心功能
    树莓派通过串口直接发送舵机控制命令给ESP32
    """
    
    def __init__(self, yolov5_path, weight_path, model_type='onnx', 
                 serial_port='COM3', baudrate=115200, 
                 track_class='down', center_threshold=50, enable_serial=True):
        """
        初始化跟踪器
        
        Args:
            yolov5_path (str): YOLOv5路径
            weight_path (str): 模型权重路径
            model_type (str): 模型类型 ('pt' 或 'onnx')
            serial_port (str): 串口号
            baudrate (int): 波特率
            track_class (str): 跟踪的目标类别 ('normal', 'down', 'both')
            center_threshold (int): 中心区域阈值
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.track_class = track_class
        self.center_threshold = center_threshold
        self.enable_serial = enable_serial
        
        # 初始化跌倒检测器
        self.detector = FallDownDetectYolo(
            yolov5_path=yolov5_path,
            weight_path=weight_path,
            image_path='',
            pt_or_onnx=model_type,
            video_path=None
        )
        
        # 检查GPU状态
        if model_type == 'pt':
            if torch.cuda.is_available():
                print(f"✅ GPU加速已启用: {torch.cuda.get_device_name(0)}")
                print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("⚠️ GPU不可用，使用CPU模式")
        else:
            print("ℹ️ ONNX模式，GPU加速取决于ONNX Runtime配置")
        
        # 初始化串口连接
        self.serial = None
        if self.enable_serial:
            self.init_serial()
        else:
            print("串口通信已禁用，仅进行检测和显示")
        
        # 跟踪状态
        self.tracking_active = False
        self.frame_size = (640, 480)
        self.last_command_time = 0  # 上次发送命令的时间
        self.command_interval = 0.5  # 命令发送间隔（秒）- 增加到0.5秒
        self.last_command = None  # 上次发送的命令，避免重复发送
        
        print(f"跟踪功能已启用，跟踪目标: {track_class}")
    
    def init_serial(self):
        """初始化串口连接"""
        try:
            self.serial = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1
            )
            print(f"串口连接成功: {self.serial_port}")
            
            # 等待ESP32初始化
            time.sleep(2)
            
        except Exception as e:
            print(f"串口连接失败: {e}")
            print("请检查串口号和连接")
            self.serial = None
    
    def send_command(self, command):
        """
        发送命令到ESP32
        
        Args:
            command (str): 命令字符串
        """
        if self.serial is None:
            return
        
        # 检查命令发送间隔
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return  # 跳过过于频繁的命令
        
        # 避免重复发送相同命令
        if command == self.last_command:
            return
        
        try:
            # 发送命令
            self.serial.write((command + '\n').encode())
            self.serial.flush()  # 确保数据立即发送
            self.last_command_time = current_time
            self.last_command = command
            
            # 打印发送的命令
            print(f"发送命令: {command}")
            
            # 短暂等待，让ESP32处理命令
            time.sleep(0.01)
            
        except Exception as e:
            print(f"发送命令失败: {e}")
    
    def calculate_target_center(self, detection_df):
        """
        计算检测目标的中心位置
        
        Args:
            detection_df: 检测结果DataFrame
            
        Returns:
            tuple: (x, y) 目标中心坐标，如果没有目标则返回None
        """
        if len(detection_df) == 0:
            return None
        
        # 根据跟踪类别筛选目标
        if self.track_class == 'both':
            target_detections = detection_df
        else:
            target_detections = detection_df[detection_df['class_name'] == self.track_class]
        
        if len(target_detections) == 0:
            return None
        
        # 选择置信度最高的目标
        best_detection = target_detections.loc[target_detections['confidence'].idxmax()]
        
        # 计算目标中心
        x_center = int((best_detection['xmin'] + best_detection['xmax']) / 2)
        y_center = int((best_detection['ymin'] + best_detection['ymax']) / 2)
        
        return (x_center, y_center)
    
    def calculate_servo_movement(self, target_center):
        """
        根据目标位置计算舵机移动命令
        
        Args:
            target_center (tuple): 目标中心位置 (x, y)
            
        Returns:
            str: 舵机移动命令
        """
        if target_center is None:
            return None
        
        frame_center_x = self.frame_size[0] // 2
        frame_center_y = self.frame_size[1] // 2
        
        target_x, target_y = target_center
        
        # 计算目标与中心的偏移
        offset_x = target_x - frame_center_x
        offset_y = target_y - frame_center_y
        
        # 如果目标在中心区域内，不需要移动
        if abs(offset_x) < self.center_threshold and abs(offset_y) < self.center_threshold:
            return None
        
        # 计算舵机角度调整
        pan_delta = int(offset_x * 0.1)  # 水平移动映射
        tilt_delta = int(offset_y * 0.1)  # 垂直移动映射
        
        # 限制移动步长
        pan_delta = max(-5, min(5, pan_delta))
        tilt_delta = max(-5, min(5, tilt_delta))
        
        # 生成移动命令
        command = f"MOVE {pan_delta} {tilt_delta}"
        
        return command
    
    def update_tracking(self, detection_df):
        """
        更新跟踪状态并发送控制命令
        
        Args:
            detection_df: 检测结果DataFrame
        """
        # 计算目标中心
        target_center = self.calculate_target_center(detection_df)
        
        if target_center is not None:
            # 找到目标，计算舵机移动命令
            command = self.calculate_servo_movement(target_center)
            
            if command:
                # 发送移动命令
                self.send_command(command)
                if not self.tracking_active:
                    self.tracking_active = True
                    print("开始跟踪目标")
            else:
                # 目标在中心，无需移动
                if self.tracking_active:
                    self.tracking_active = False
                    print("目标已居中")
                    # 发送停止命令
                    self.send_command("STOP")
            
        else:
            # 目标丢失
            if self.tracking_active:
                print("目标丢失")
                self.tracking_active = False
                # 发送停止命令
                self.send_command("STOP")
    
    def init_camera(self, camera_index=0, camera_api='dshow'):
        """
        初始化摄像头，采用更稳定的方式
        
        Args:
            camera_index (int): 摄像头索引
            
        Returns:
            cv2.VideoCapture: 摄像头对象，失败返回None
        """
        import cv2
        
        # 根据参数选择API
        api_map = {
            'dshow': cv2.CAP_DSHOW,
            'any': cv2.CAP_ANY,
            'msmf': cv2.CAP_MSMF
        }
        
        # 尝试不同的摄像头设置
        camera_configs = [
            {'index': camera_index, 'api': api_map.get(camera_api, cv2.CAP_DSHOW)},
            {'index': camera_index, 'api': cv2.CAP_ANY},
            {'index': 0, 'api': cv2.CAP_DSHOW},
            {'index': 1, 'api': cv2.CAP_DSHOW},
        ]
        
        cap = None
        
        for config in camera_configs:
            try:
                print(f"尝试打开摄像头: 索引={config['index']}, API={config['api']}")
                cap = cv2.VideoCapture(config['index'], config['api'])
                
                if cap.isOpened():
                    # 设置摄像头参数以提高稳定性
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # 等待摄像头初始化
                    time.sleep(2)
                    
                    # 尝试读取多帧来确认摄像头工作正常
                    success_count = 0
                    for _ in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            success_count += 1
                        time.sleep(0.1)
                    
                    if success_count >= 3:  # 至少3帧成功
                        print(f"✅ 摄像头初始化成功: 索引={config['index']}, API={config['api']}")
                        print(f"   分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"   帧率: {cap.get(cv2.CAP_PROP_FPS):.1f}")
                        break
                    else:
                        print(f"❌ 摄像头索引 {config['index']} 读取不稳定")
                        cap.release()
                        cap = None
                else:
                    print(f"❌ 无法打开摄像头索引 {config['index']}")
                    if cap:
                        cap.release()
                        cap = None
                        
            except Exception as e:
                print(f"❌ 摄像头初始化失败: {e}")
                if cap:
                    cap.release()
                    cap = None
                continue
        
        if cap is None:
            print("❌ 所有摄像头初始化尝试都失败了")
            print("💡 建议:")
            print("   1. 检查摄像头是否被其他程序占用")
            print("   2. 尝试重启摄像头或电脑")
            print("   3. 检查摄像头驱动")
            return None
            
        return cap
    
    def run_video_tracking(self, fps=15, camera_index=0, camera_api='dshow'):
        """
        运行视频跟踪
        
        Args:
            fps (int): 帧率
            camera_index (int): 摄像头索引
            camera_api (str): 摄像头API
        """
        import cv2
        
        # 禁用OpenCV警告（兼容不同版本）
        try:
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        except AttributeError:
            # 旧版本OpenCV不支持setLogLevel，忽略错误
            pass
        
        # 检查串口连接（如果启用串口）
        if self.enable_serial and self.serial is None:
            print("串口未连接，无法运行跟踪")
            return
        
        # 初始化摄像头
        print("正在初始化摄像头...")
        cap = self.init_camera(camera_index, camera_api)
        
        if cap is None:
            print("无法初始化摄像头")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (width, height)
        
        print(f"视频分辨率: {width}x{height}, FPS: {fps}")
        print("开始视频跟踪，按 'q' 键退出...")
        
        # GPU性能监控
        if self.detector.pt_or_onnx == 'pt' and torch.cuda.is_available():
            print("🔍 GPU性能监控已启用")
        
        frame_count = 0
        start_time = time.time()
        
        consecutive_failures = 0
        max_failures = 5
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"⚠️ 读取帧失败 ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("❌ 连续读取失败次数过多，尝试重新初始化摄像头...")
                    cap.release()
                    time.sleep(3)
                    cap = self.init_camera()
                    if cap is None:
                        print("❌ 重新初始化摄像头失败，退出程序")
                        break
                    consecutive_failures = 0
                    continue
                
                time.sleep(0.1)  # 短暂等待
                continue
            
            # 重置失败计数
            consecutive_failures = 0
            frame_count += 1
            
            # 根据模型类型进行预处理和推理
            if self.detector.pt_or_onnx == 'pt':
                # GPU推理优化
                with torch.no_grad():  # 禁用梯度计算以提高推理速度
                    results = self.detector.model(frame)

                # 将检测结果转换为 Pandas DataFrame
                df = results.pandas().xyxy[0]

                # 添加类别名称
                df['class_name'] = df['class'].apply(lambda x: self.detector.class_names[int(x)])

            elif self.detector.pt_or_onnx == 'onnx':
                # 预处理
                preprocessed_img, ratio, padding = self.detector.preprocess_image(frame)

                # 进行推理
                outputs = self.detector.session.run(self.detector.output_names, {self.detector.input_name: preprocessed_img})[0]

                # 后处理
                df = self.detector.postprocess_image(outputs, ratio, padding)

            # 输出检测结果（每10帧输出一次）
            if frame_count % 10 == 0:
                if len(df) > 0:
                    # 获取置信度最高的检测结果
                    best_detection = df.loc[df['confidence'].idxmax()]
                    class_name = best_detection['class_name']
                    confidence = best_detection['confidence']
                    print(f"帧 {frame_count}: 检测到 {class_name} (置信度: {confidence:.3f})")
                else:
                    print(f"帧 {frame_count}: 未检测到目标")
                
                # GPU内存监控
                if self.detector.pt_or_onnx == 'pt' and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                    print(f"GPU内存使用: {gpu_memory:.1f} MB")

            # 更新跟踪
            self.update_tracking(df)

            # 绘制检测框
            detected_frame = self.detector.draw_detections(frame.copy(), df)

            # 显示检测结果
            cv2.imshow('Fall Detection Tracking', detected_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户按下了 'q' 键，退出视频跟踪")
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        if self.serial:
            self.serial.close()
        
        # 性能统计
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n🎯 性能统计:")
        print(f"总处理帧数: {frame_count}")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"平均FPS: {avg_fps:.2f}")
        
        # GPU内存清理
        if self.detector.pt_or_onnx == 'pt' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU内存已清理")
        
        print(f"\n视频跟踪完成，共处理 {frame_count} 帧")


def main():
    """主函数"""
    # 获取当前脚本所在目录的上级目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    default_yolov5_path = os.path.join(project_root, "yolov5")
    default_weight_path = os.path.join(project_root, "modelweight", "yolov5_best.pt")
    
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description="简化的目标跟踪器 - 无UI版本",
        epilog="示例: python simple_tracking.py --serial_port COM3 --track_class down"
    )
    
    # 添加参数
    parser.add_argument("--yolov5_path", type=str, default=default_yolov5_path, 
                       help="YOLOv5 本地路径")
    parser.add_argument("--weight_path", type=str, default=default_weight_path, 
                       help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--model_type", type=str, default='pt', choices=['pt', 'onnx'], 
                       help="模型类型：pt (推荐GPU) 或 onnx")
    parser.add_argument("--serial_port", type=str, default='COM3', 
                       help="串口号")
    parser.add_argument("--baudrate", type=int, default=115200, 
                       help="波特率")
    parser.add_argument("--track_class", type=str, default='down', 
                       choices=['normal', 'down', 'both'], help="跟踪的目标类别")
    parser.add_argument("--center_threshold", type=int, default=50, 
                       help="中心区域阈值")
    parser.add_argument("--fps", type=int, default=15, 
                       help="视频帧率")
    parser.add_argument("--no_serial", action='store_true', 
                       help="禁用串口通信，仅进行检测和显示")
    parser.add_argument("--camera_index", type=int, default=0, 
                       help="摄像头索引 (默认: 0)")
    parser.add_argument("--camera_api", type=str, default='dshow', 
                       choices=['dshow', 'any', 'msmf'], 
                       help="摄像头API (默认: dshow)")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = SimpleTracking(
        yolov5_path=args.yolov5_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        track_class=args.track_class,
        center_threshold=args.center_threshold,
        enable_serial=not args.no_serial
    )
    
    # 运行视频跟踪
    tracker.run_video_tracking(
        fps=args.fps,
        camera_index=args.camera_index,
        camera_api=args.camera_api
    )


if __name__ == "__main__":
    main() 