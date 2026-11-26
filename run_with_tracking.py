import argparse
import os
import sys
import time
from fallDownDetectYolo import FallDownDetectYolo
from servo_controller import ServoController, MockServoController

class FallDetectionWithTracking:
    """
    集成舵机跟踪功能的跌倒检测器
    """
    
    def __init__(self, yolov5_path, weight_path, model_type='onnx', 
                 servo_control_type='mock', servo_port='COM3', 
                 enable_tracking=True, track_class='down'):
        """
        初始化跌倒检测跟踪器
        
        Args:
            yolov5_path (str): YOLOv5路径
            weight_path (str): 模型权重路径
            model_type (str): 模型类型 ('pt' 或 'onnx')
            servo_control_type (str): 舵机控制类型 ('serial', 'pwm', 'mock')
            servo_port (str): 串口号
            enable_tracking (bool): 是否启用跟踪功能
            track_class (str): 跟踪的目标类别 ('normal', 'down', 'both')
        """
        self.yolov5_path = yolov5_path
        self.weight_path = weight_path
        self.model_type = model_type
        self.enable_tracking = enable_tracking
        self.track_class = track_class
        
        # 初始化跌倒检测器
        self.detector = FallDownDetectYolo(
            yolov5_path=yolov5_path,
            weight_path=weight_path,
            image_path='',  # 视频模式不需要图片路径
            pt_or_onnx=model_type,
            video_path=None  # 使用摄像头
        )
        
        # 初始化舵机控制器
        if enable_tracking:
            if servo_control_type == 'mock':
                self.servo_controller = MockServoController()
            else:
                self.servo_controller = ServoController(
                    control_type=servo_control_type,
                    port=servo_port
                )
            print(f"舵机跟踪功能已启用，跟踪目标: {track_class}")
        else:
            self.servo_controller = None
            print("舵机跟踪功能已禁用")
        
        # 跟踪状态
        self.tracking_active = False
        self.last_target_center = None
        self.target_lost_frames = 0
        self.max_target_lost_frames = 30  # 目标丢失30帧后停止跟踪
        
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
    
    def update_tracking(self, detection_df, frame_size):
        """
        更新跟踪状态
        
        Args:
            detection_df: 检测结果DataFrame
            frame_size: 视频帧大小 (width, height)
        """
        if not self.enable_tracking or self.servo_controller is None:
            return
        
        # 设置帧大小
        self.servo_controller.set_frame_size(*frame_size)
        
        # 计算目标中心
        target_center = self.calculate_target_center(detection_df)
        
        if target_center is not None:
            # 找到目标，更新跟踪状态
            self.last_target_center = target_center
            self.target_lost_frames = 0
            self.tracking_active = True
            
            # 执行舵机跟踪
            self.servo_controller.track_target(target_center)
            
        else:
            # 目标丢失
            self.target_lost_frames += 1
            
            if self.target_lost_frames > self.max_target_lost_frames:
                # 目标丢失时间过长，停止跟踪
                if self.tracking_active:
                    print("目标丢失时间过长，停止跟踪")
                    self.tracking_active = False
                    # 可以选择重置舵机位置
                    # self.servo_controller.reset_position()
    
    def run_video_inference_with_tracking(self, save=False, save_path='output_with_tracking.mp4', fps=30):
        """
        运行带跟踪功能的视频推理
        
        Args:
            save (bool): 是否保存视频
            save_path (str): 保存路径
            fps (int): 帧率
        """
        import cv2
        
        # 打开摄像头
        print("正在打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        
        print(f"视频分辨率: {width}x{height}, FPS: {fps}")
        
        # 初始化视频写入器
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
            print(f"检测结果视频将保存至: {save_path}")
        
        frame_count = 0
        print("开始视频推理和跟踪，按 'q' 键退出...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"\n无法读取第 {frame_count + 1} 帧，视频可能已结束")
                    break
                
                frame_count += 1
                
                # 进行目标检测
                if self.model_type == 'pt':
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
                self.update_tracking(df, frame_size)
                
                # 绘制检测框和跟踪信息
                detected_frame = self.detector.draw_detections(frame.copy(), df)
                
                # 绘制跟踪状态信息
                self.draw_tracking_info(detected_frame)
                
                # 显示结果
                cv2.imshow('Fall Detection with Tracking', detected_frame)
                
                # 保存视频
                if save:
                    out.write(detected_frame)
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户按下了 'q' 键，退出视频推理")
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
            
            if self.servo_controller:
                self.servo_controller.close()
            
            print(f"\n视频推理完成，共处理 {frame_count} 帧")
    
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
        
        # 绘制舵机位置信息
        if self.servo_controller:
            servo_text = f"Servo: PAN={self.servo_controller.current_pan}, TILT={self.servo_controller.current_tilt}"
            cv2.putText(frame, servo_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)
        
        # 绘制目标丢失计数
        if self.target_lost_frames > 0:
            lost_text = f"Target Lost: {self.target_lost_frames} frames"
            cv2.putText(frame, lost_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 165, 255), 2)


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
        description="带舵机跟踪功能的跌倒检测器",
        epilog="示例: python run_with_tracking.py --track_class down --servo_type serial --servo_port COM3"
    )
    
    # 添加参数
    parser.add_argument("--yolov5_path", type=str, default=default_yolov5_path, 
                       help="YOLOv5 本地路径")
    parser.add_argument("--weight_path", type=str, default=default_weight_path, 
                       help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--model_type", type=str, default='onnx', choices=['pt', 'onnx'], 
                       help="模型类型：pt 或 onnx")
    parser.add_argument("--servo_type", type=str, default='mock', 
                       choices=['serial', 'pwm', 'mock'], help="舵机控制类型")
    parser.add_argument("--servo_port", type=str, default='COM3', 
                       help="舵机串口号")
    parser.add_argument("--track_class", type=str, default='down', 
                       choices=['normal', 'down', 'both'], help="跟踪的目标类别")
    parser.add_argument("--disable_tracking", action='store_true', 
                       help="禁用跟踪功能")
    parser.add_argument("--save", action='store_true', 
                       help="是否保存检测结果视频")
    parser.add_argument("--save_path", type=str, default='output_with_tracking.mp4', 
                       help="检测结果保存路径")
    parser.add_argument("--fps", type=int, default=30, 
                       help="视频帧率")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建检测跟踪器
    detector = FallDetectionWithTracking(
        yolov5_path=args.yolov5_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        servo_control_type=args.servo_type,
        servo_port=args.servo_port,
        enable_tracking=not args.disable_tracking,
        track_class=args.track_class
    )
    
    # 运行视频推理和跟踪
    detector.run_video_inference_with_tracking(
        save=args.save,
        save_path=args.save_path,
        fps=args.fps
    )


if __name__ == "__main__":
    main() 