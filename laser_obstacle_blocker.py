import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import serial

class LaserObstacleBlocker(Node):
    def __init__(self):
        super().__init__('laser_obstacle_blocker')

        # === 参数设置 ===
        self.serial_port = '/dev/ttyUSB1'  # ESP32 串口端口
        self.baudrate = 115200             # 波特率
        self.response_dist = 0.5           # 避障距离阈值（单位：米）
        self.angle_range = 10              # 前方角度范围 ±10°
        self.last_cmd = None
        self.blocking = False

        # === 串口打开 ===
        try:
            self.esp = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
            self.get_logger().info(f"串口已连接 {self.serial_port}")
        except Exception as e:
            self.get_logger().error(f"串口打开失败: {e}")
            exit(1)

        # === 订阅雷达话题 ===
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, scan_data):
        ranges = np.array(scan_data.ranges)
        angles = np.arange(len(ranges)) * scan_data.angle_increment + scan_data.angle_min
        angles_deg = angles * 180.0 / np.pi

        front_indices = np.where(np.abs(angles_deg) < self.angle_range)[0]
        front_distances = ranges[front_indices]

        if np.any(front_distances < self.response_dist):
            if self.last_cmd != 'S':
                self.esp.write(b'S\n')
                self.get_logger().warn("前方小于50cm，发送：S（停车）")
                self.last_cmd = 'S'
                self.blocking = True
        else:
            if self.blocking:
                self.get_logger().info("障碍清除，等待 UWB 控制")
                self.blocking = False
                self.last_cmd = None

def main(args=None):
    rclpy.init(args=args)
    node = LaserObstacleBlocker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
