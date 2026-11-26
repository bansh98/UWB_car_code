import argparse
import os
import sys
import time
import serial
from fallDownDetectYolo import FallDownDetectYolo

class SerialTracking:
    """
    串口通信的目标跟踪器
    树莓派通过串口直接发送舵机控制命令给ESP32
    """
    
    def __init__(self, yolov5_path, weight_path, model_type='onnx', 
                 serial_port='COM3', baudrate=115200, 
                 track_class='down', center_threshold=50):
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
        
        # 初始化跌倒检测器
        self.detector = FallDownDetectYolo(
            yolov5_path=yolov5_path,
            weight_path=weight_path,
            image_path='',
            pt_or_onnx=model_type,
            video_path=None
        )
        
        # 初始化串口连接
        self.serial = None
        self.init_serial()
        
        # 跟踪状态
        self.tracking_active = False
        self.frame_size = (640, 480)
        
        print(f"串口跟踪功能已启用，跟踪目标: {track_class}")
    
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
            
            # 发送测试命令
            self.send_command("STATUS")
            
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
            print("串口未连接")
            return
        
        try:
            # 发送命令
            self.serial.write((command + '\n').encode())
            
            # 读取响应
            response = self.serial.readline().decode().strip()
            if response:
                print(f"ESP32响应: {response}")
                
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
                self.tracking_active = True
            else:
                # 目标在中心，无需移动
                if self.tracking_active:
                    print("目标在中心，无需移动")
                    self.tracking_active = False
            
        else:
            # 目标丢失
            if self.tracking_active:
                print("目标丢失")
                self.tracking_active = False
    
    def run_video_tracking(self, save=False, save_path='tracking_output.mp4', fps=30):
        """
        运行视频跟踪
        
        Args:
            save (bool): 是否保存视频
            save_path (str): 保存路径
            fps (int): 帧率
        """
        import cv2
        
        # 检查串口连接
        if self.serial is None:
            print("串口未连接，无法运行跟踪")
            return
        
        # 打开摄像头
        print("正在打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (width, height)
        
        print(f"视频分辨率: {width}x{height}, FPS: {fps}")
        
        # 初始化视频写入器
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, fps, self.frame_size)
            print(f"检测结果视频将保存至: {save_path}")
        
        frame_count = 0
        print("开始视频跟踪，按 'q' 键退出...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"\n无法读取第 {frame_count + 1} 帧，视频可能已结束")
                    break
                
                frame_count += 1
                
                # 进行目标检测
                if self.detector.pt_or_onnx == 'pt':
                    results = self.detector.model(frame)
                    df = results.pandas().xyxy[0]
                    df['class_name'] = df['class'].apply(lambda x: self.detector.class_names[int(x)])
                else:  # onnx
                    preprocessed_img, ratio, padding = self.detector.preprocess_image(frame)
                    outputs = self.detector.session.run(self.detector.output_names, 
                                                      {self.detector.input_name: preprocessed_img})[0]
                    df = self.detector.postprocess_image(outputs, ratio, padding)
                
                # 输出检测结果
                if len(df) > 0:
                    best_detection = df.loc[df['confidence'].idxmax()]
                    class_name = best_detection['class_name']
                    confidence = best_detection['confidence']
                    print(f"帧 {frame_count}: 检测到 {class_name} (置信度: {confidence:.3f})")
                else:
                    print(f"帧 {frame_count}: 未检测到目标")
                
                # 更新跟踪
                self.update_tracking(df)
                
                # 绘制检测框和跟踪信息
                detected_frame = self.detector.draw_detections(frame.copy(), df)
                
                # 绘制跟踪状态信息
                self.draw_tracking_info(detected_frame)
                
                # 显示结果
                cv2.imshow('Serial Tracking', detected_frame)
                
                # 保存视频
                if save:
                    out.write(detected_frame)
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户按下了 'q' 键，退出视频跟踪")
                    break
                
                # 控制帧率
                time.sleep(1/fps)
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        
        finally:
            # 释放资源
            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()
            
            if self.serial:
                self.serial.close()
            
            print(f"\n视频跟踪完成，共处理 {frame_count} 帧")
    
    def draw_tracking_info(self, frame):
        """
        在帧上绘制跟踪状态信息
        
        Args:
            frame: 要绘制的帧
        """
        import cv2
        
        # 绘制跟踪状态
        status_text = f"Tracking: {'ON' if self.tracking_active else 'OFF'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0) if self.tracking_active else (0, 0, 255), 2)
        
        # 绘制目标类别
        class_text = f"Target: {self.track_class}"
        cv2.putText(frame, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # 绘制串口连接信息
        serial_text = f"Serial: {self.serial_port}"
        cv2.putText(frame, serial_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)


def main():
    """主函数"""
    # 获取当前脚本所在目录的上级目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    default_yolov5_path = os.path.join(project_root, "yolov5")
    default_weight_path = os.path.join(project_root, "modelweight", "model.onnx")
    
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description="串口通信的目标跟踪器",
        epilog="示例: python serial_tracking.py --serial_port COM3 --track_class down"
    )
    
    # 添加参数
    parser.add_argument("--yolov5_path", type=str, default=default_yolov5_path, 
                       help="YOLOv5 本地路径")
    parser.add_argument("--weight_path", type=str, default=default_weight_path, 
                       help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--model_type", type=str, default='onnx', choices=['pt', 'onnx'], 
                       help="模型类型：pt 或 onnx")
    parser.add_argument("--serial_port", type=str, default='COM3', 
                       help="串口号")
    parser.add_argument("--baudrate", type=int, default=115200, 
                       help="波特率")
    parser.add_argument("--track_class", type=str, default='down', 
                       choices=['normal', 'down', 'both'], help="跟踪的目标类别")
    parser.add_argument("--center_threshold", type=int, default=50, 
                       help="中心区域阈值")
    parser.add_argument("--save", action='store_true', 
                       help="是否保存检测结果视频")
    parser.add_argument("--save_path", type=str, default='tracking_output.mp4', 
                       help="检测结果保存路径")
    parser.add_argument("--fps", type=int, default=30, 
                       help="视频帧率")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = SerialTracking(
        yolov5_path=args.yolov5_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        track_class=args.track_class,
        center_threshold=args.center_threshold
    )
    
    # 运行视频跟踪
    tracker.run_video_tracking(
        save=args.save,
        save_path=args.save_path,
        fps=args.fps
    )


if __name__ == "__main__":
    main() 