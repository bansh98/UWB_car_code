import serial
import time
import threading
import math

PORTS = {
    "TB": "/dev/ttyACM1",  # TB模式（test1.py 逻辑）you
    "TA": "/dev/ttyACM0"   # TA模式（testta.py 逻辑）zuo
}
BAUDRATE = 115200
FRAME_HEADER = b'\x43\x6D\x64'
FRAME_SIZE = 64

# AB 实际距离（单位：米）
AB_DISTANCE = 0.34

# ESP32 控制串口
ESP32_PORT = '/dev/ttyUSB0'
ESP32_BAUDRATE = 115200
esp32_serial = None  # 全局串口对象

# 全局变量保存最新距离
global_latest_ta = None  # TA = ra
global_latest_tb = None  # TB = rb
# 距离历史，用于判断是否停车
ts_history = []

# 新增角度调整常量
ROTATION_FACTOR = 0.00556  # 2秒/360度
TARGET_ANGLE = 90          # 目标角度
ANGLE_THRESHOLD = 5        # 角度阈值

def find_valid_frame(buffer: bytearray) -> bytes:
    start = buffer.find(FRAME_HEADER)
    if start != -1 and len(buffer) >= start + FRAME_SIZE:
        frame = buffer[start:start + FRAME_SIZE]
        del buffer[:start + FRAME_SIZE]
        return frame
    return None

def parse_distance_tb(frame: bytes) -> float:
    if len(frame) < 26:
        return None
    dist_bytes = frame[21:25]
    dist_mm = int.from_bytes(dist_bytes, byteorder='little')
    return dist_mm / 1000.0

def parse_distance_ta(frame: bytes) -> float:
    if frame[0:3] != FRAME_HEADER:
        return None
    dist_mm = frame[17] + (frame[18] << 8)
    return dist_mm / 1000.0

def print_hex(frame: bytes):
    hex_str = ' '.join(f'{b:02X}' for b in frame)
    print(f"[HEX] {hex_str}")

def compute_angle_and_judge(ra, rb, d):
    try:
        if rb == 0 or d == 0:
            return None, "参数错误"
        cos_val = (rb**2 + d**2 - ra**2) / (2 * rb * d)
        cos_val = max(-1.0, min(1.0, cos_val))
        angle_rad = math.acos(cos_val)
        angle_deg = math.degrees(angle_rad)
        if 85 <= angle_deg <= 95:
            result = "小车对准目标，直行"
        elif 0 < angle_deg < 85:
            result = f"目标偏左{90-angle_deg:.1f}度，需要左转"
        elif 95 < angle_deg < 180:
            result = f"目标偏右{angle_deg-90:.1f}度，需要右转"
        else:
            result = "角度异常"
        return angle_deg, result
    except Exception as e:
        return None, f"计算错误: {e}"

def compute_ts_distance(ra, rb, d):
    try:
        x = (ra**2 - rb**2 + d**2) / (2 * d)
        y_sq = ra**2 - x**2
        if y_sq < 0:
            return None, "无效（y^2<0，三角形不成立）"
        y = y_sq**0.5
        sx = d / 2
        sy = 0
        ts = ((x - sx)**2 + (y - sy)**2)**0.5
        return ts, None
    except Exception as e:
        return None, f"计算错误: {e}"

def decide_control_command(ts, angle, ts_history=None):
    """
    ts: 当前TS距离
    angle: 当前角度（度）
    ts_history: 距离历史列表（用于判断是否停车）
    返回: 控制指令字符串
    """
    # 角度异常情况判断
    if angle is None or angle < 0 or angle > 180:
        return 'S'
    
    # 距离优先判断 - 距离过近时后退
    if ts < 0.5:
        return 'B'
    
    # 角度偏差计算
    angle_diff = angle - TARGET_ANGLE
    
    # 角度调整判断
    if abs(angle_diff) <= ANGLE_THRESHOLD:
        # 角度正确，前进
        return 'F'
    elif angle_diff > ANGLE_THRESHOLD:
        # 右转
        rotation_time = abs(angle_diff) * ROTATION_FACTOR
        return f'R_{rotation_time:.3f}'
    elif angle_diff < -ANGLE_THRESHOLD:
        # 左转
        rotation_time = abs(angle_diff) * ROTATION_FACTOR
        return f'L_{rotation_time:.3f}'
    
    # 兜底 - 默认前进
    return 'F'

def send_cmd_to_esp32(cmd):
    global esp32_serial
    if esp32_serial is not None:
        try:
            esp32_serial.write((cmd + '\n').encode())
        except Exception as e:
            print(f"[ESP32串口] 发送失败: {e}")

def update_and_compute(mode, dist):
    global global_latest_ta, global_latest_tb, ts_history
    if mode == "TA":
        global_latest_ta = dist
    elif mode == "TB":
        global_latest_tb = dist
    if global_latest_ta is not None and global_latest_tb is not None:
        angle, result = compute_angle_and_judge(global_latest_ta, global_latest_tb, AB_DISTANCE)
        if angle is not None:
            print(f"【角度计算】TA={global_latest_ta:.2f} m, TB={global_latest_tb:.2f} m, ∠TSB={angle:.2f}°，{result}")
        ts, ts_err = compute_ts_distance(global_latest_ta, global_latest_tb, AB_DISTANCE)
        if ts is not None and angle is not None:
            print(f"【TS距离】T到S的距离 = {ts:.3f} m")
            ts_history.append(ts)
            if len(ts_history) > 10:
                ts_history.pop(0)
            # 只在角度不在80-120度时才旋转
            if angle < 80 or angle > 120:
                angle_diff = angle - TARGET_ANGLE
                cmd = 'R' if angle_diff > 0 else 'L'
                rotation_time = abs(angle_diff) * ROTATION_FACTOR
                print(f"【控制指令】{cmd}，旋转时间{rotation_time:.3f}s")
                send_cmd_to_esp32(cmd)
                time.sleep(rotation_time)
                send_cmd_to_esp32('S')
                time.sleep(0.1)  # 稍作停顿
            else:
                # 角度在80-120度，直接距离判断
                if ts < 0.5:
                    cmd = 'B'
                else:
                    cmd = 'F'
                print(f"【控制指令】{cmd}")
                send_cmd_to_esp32(cmd)
        else:
            print(f"【TS距离】计算失败：{ts_err}")
            print(f"【控制指令】S")
            send_cmd_to_esp32('S')

def read_uwb(port, mode):
    try:
        ser = serial.Serial(port, BAUDRATE, timeout=0.1)
    except Exception as e:
        print(f"[{mode}] 打开串口失败: {e}")
        return
    buffer = bytearray()
    while True:
        data = ser.read(128)
        if data:
            buffer.extend(data)
            while True:
                frame = find_valid_frame(buffer)
                if not frame:
                    break
                # print_hex(frame)  # 如需调试可打开
                if mode == "TB":
                    dist = parse_distance_tb(frame)
                else:
                    dist = parse_distance_ta(frame)
                if dist is not None:
                    print(f"[{mode}] 距离 = {dist:.2f} m")
                    update_and_compute(mode, dist)
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        esp32_serial = serial.Serial(ESP32_PORT, ESP32_BAUDRATE, timeout=0.1)
        print(f"已打开ESP32串口: {ESP32_PORT}")
    except Exception as e:
        print(f"[警告] 无法打开ESP32串口: {e}")
        esp32_serial = None
    threads = []
    for mode, port in PORTS.items():
        t = threading.Thread(target=read_uwb, args=(port, mode), daemon=True)
        t.start()
        threads.append(t)
    print("同时监听 TA (acm6) 和 TB (acm5) ... 按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[退出] 已停止读取")