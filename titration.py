"""Mlabs AI Titration System v1.5 - 基于计算机视觉的自动滴定实验系统"""

import json
import os
import re
import shutil
import time
import traceback
import warnings
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import serial
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.optimize import curve_fit

import Find_COM
from model import resnet34


PLATFORM_BASE_URL = "https://jingsai.mools.net"
LOGIN_URL = f"{PLATFORM_BASE_URL}/api/login"
UPLOAD_URL = f"{PLATFORM_BASE_URL}/api/upload-record"

REQUIRED_DIRS = ["Input", "Output", "pths"]


def format_date_time(date_time_str):
    dt = datetime.strptime(date_time_str, "%Y%m%d_%H%M%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_environment():
    for folder in REQUIRED_DIRS:
        if not os.path.exists(folder):
            os.makedirs(folder)


def login_to_platform(username, password):
    try:
        payload = {"userName": username, "password": password}
        response = requests.post(LOGIN_URL, payload, timeout=30)
        if response is None:
            return None
        result = json.loads(response.text)
        if result.get("code") == 1:
            return result["token"]
        else:
            print(f"登录失败: {result.get('msg', '未知错误')}")
            return None
    except Exception as e:
        print(f"登录时出错: {e}")
        return None


def send_data_to_platform(token, data, picture):
    try:
        if not picture or not os.path.exists(picture):
            print("图片文件不存在，跳过上传")
            return False
        with open(picture, "rb") as picture_file:
            base64_encoded_picture = base64.b64encode(picture_file.read()).decode("utf-8")
        data["final_image"] = base64_encoded_picture
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        json_data = json.dumps(data, ensure_ascii=False)
        response = requests.post(UPLOAD_URL, headers=headers, data=json_data.encode("utf-8"), timeout=30)
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get("code") == 1:
                print("上传成功！")
                return True
        return False
    except Exception as e:
        print(f"发送数据时出错: {e}")
        return False


class MATConfig:
    RESIZE_SIZE = 256
    CROP_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    SERIAL_BAUDRATE = 115200
    PUMP_STATUS_TIMEOUT = 0.5
    MOVE_TIMEOUT = 30
    VOLTAGE_TIMEOUT = 2

    DEFAULT_FINAL_VOLUME = 100000000
    DRAW_VOLUME = 20
    VOLUME_DECIMALS = 3

    WB_TARGET = 128
    WB_FACTOR = 1.1


class MAT:
    def __init__(self, video_source_index=0, model_name="resnet34-1", classes=2, overdose=0, volume_par=1.0,
                 final_volume=0, typ="All", transition_class=None):
        self.last_prob = None
        self.last_class = None
        print("实验初始化中")
        self.type = typ
        self.data_root = os.getcwd()
        self.video_source_index = video_source_index

        self.cap = cv2.VideoCapture(video_source_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.port = Find_COM.find_any_port()
        if self.port:
            self.pump_ser = serial.Serial(self.port, MATConfig.SERIAL_BAUDRATE)
        else:
            print("警告: 未检测到串口设备，注射泵功能不可用")
            self.pump_ser = None

        self.volume_par = volume_par
        self.classes = classes
        self.final_volume = final_volume if final_volume else MATConfig.DEFAULT_FINAL_VOLUME
        self.total_volume = 0
        self.now_volume = 0
        self.volume_list = []
        self.voltage_list = []
        self.color_list = []
        self.start_time = None
        self.formatted_start_time = None
        self.finish_volume = 0
        self.finish_picture = ""

        self.overdose = overdose
        self.transition_class = transition_class
        self.transition_detected = False

        self.state = "FAST"
        self.smooth_window = 5
        self.blue_threshold = 0.70
        self.purple_threshold = 0.45
        self.confirm_frames = 3
        self.prob_buffer = []
        self.consecutive_blue = 0
        self.consecutive_purple = 0
        self.first_blue_volume = None
        self.slow_loop_count = 0

        self.feedback_dir = os.path.join(self.data_root, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)

        self.weights_path = os.path.join(self.data_root, "pths", f"{model_name}Net.pth")
        self.json_path = os.path.join(self.data_root, "pths", f"{model_name}.json")

        with open(self.json_path, "r") as f:
            self.class_indict = json.load(f)

        self.model = resnet34(num_classes=self.classes).to(self.device)
        assert os.path.exists(self.weights_path), f"file: '{self.weights_path}' does not exist."
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        if self.pump_ser:
            self.servo(0)
            self.rezero()
        else:
            print("跳过舵机和归零操作（无串口）")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.close_serial()
        self.close_camera()
        cv2.destroyAllWindows()
        print("Experiment finished.")

    def get_pump_status(self, flush_buffer=True, timeout=MATConfig.PUMP_STATUS_TIMEOUT):
        if not self.pump_ser:
            return None
        try:
            if flush_buffer:
                self.pump_ser.reset_input_buffer()
            start_time = time.time()
            response = ""
            while time.time() - start_time < timeout:
                data = self.pump_ser.read_all().decode()
                if data:
                    response += data
                    if "S|" in response:
                        break
                time.sleep(0.01)
            if not response:
                return None
            matches = re.findall(r"S\|([0-9A-Fa-f]+)", response, re.IGNORECASE)
            if not matches:
                print(f"未找到状态包，原始响应: {response[:200]}")
                return None
            last_match = matches[-1]
            packed_value = int(last_match, 16)
            return {
                "v_int": (packed_value >> 45) & 0x1FFFF,
                "b_int": (packed_value >> 44) & 0x1,
                "h_int": (packed_value >> 43) & 0x1,
                "p_int": (packed_value >> 27) & 0xFFFF,
                "volume_ml": ((packed_value >> 45) & 0x1FFFF) / 1000.0,
                "voltage_v": ((packed_value >> 27) & 0xFFFF),
                "is_busy": bool((packed_value >> 44) & 0x1),
                "is_homing": bool((packed_value >> 43) & 0x1),
            }
        except Exception as e:
            print(f"获取状态失败: {e}")
            return None

    def wait_for_move_complete(self, timeout=MATConfig.MOVE_TIMEOUT, check_interval=0.2):
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_pump_status()
            if status is None:
                time.sleep(check_interval)
                continue
            if not status["is_busy"]:
                print(f"注射泵运动完成 - 体积: {status['volume_ml']:.3f}ml")
                return True
            time.sleep(check_interval)
        print(f"错误: 等待移动完成超时 ({timeout}秒)")
        return False

    def get_picture(self, frame, date=""):
        if frame is None:
            return ""
        input_dir = os.path.join(self.data_root, "Input")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        image_name = f"{date}_{self.total_volume}.jpg"
        filepath = os.path.join(input_dir, image_name)
        cv2.imwrite(filepath, frame)
        return image_name

    def rezero(self, timeout=MATConfig.MOVE_TIMEOUT):
        if not self.pump_ser:
            return
        data = b"RE\n"
        self.pump_ser.write(data)
        print("发送归零命令: RE")
        time.sleep(1)
        while not self.wait_for_move_complete(timeout=timeout):
            time.sleep(0.1)

    def start_move_1(self, timeout=MATConfig.MOVE_TIMEOUT):
        if not self.pump_ser:
            return False
        data = b"IP|20,CCW\n"
        self.pump_ser.write(data)
        print("发送抽取命令: IP|20,CCW")
        time.sleep(0.5)
        return self.wait_for_move_complete(timeout=timeout)

    def start_move_2(self, speed=0.1, timeout=MATConfig.MOVE_TIMEOUT):
        if not self.pump_ser:
            return False
        data = f"IP|{speed},CW\n"
        self.pump_ser.write(data.encode("utf-8"))
        print(f"发送移动命令: IP|{speed},CW")
        time.sleep(0.5)
        return self.wait_for_move_complete(timeout=timeout)

    def servo(self, angle):
        if not self.pump_ser:
            return
        data = f"0,{angle}\n"
        self.pump_ser.write(data.encode("utf-8"))
        time.sleep(0.5)

    def voltage(self, timeout=MATConfig.VOLTAGE_TIMEOUT):
        if not self.pump_ser:
            return 0.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.pump_ser.read_all().decode().strip()
            if response:
                matches = re.findall(r"S\|([0-9A-Fa-f]+)", response, re.IGNORECASE)
                if matches:
                    last_match = matches[-1]
                    packed_value = int(last_match, 16)
                    voltage_mv = (packed_value >> 27) & 0xFFFF
                    return voltage_mv
            time.sleep(0.05)
        print(f"警告: 读取电压超时 ({timeout}秒)")
        return 0.0

    @staticmethod
    def poly_func(x, a, b, c, d):
        return a * np.tanh(d * x + b) + c

    def line_chart(self):
        if len(self.volume_list) == 0:
            print("没有数据，跳过绘图")
            return
        x = self.volume_list
        z = self.color_list if self.color_list else [0] * len(x)
        fig, ax1 = plt.subplots()
        plt.title("Titration Curve")
        has_voltage = len(self.voltage_list) > 0
        if has_voltage:
            y = self.voltage_list
            color = "tab:red"
            ax1.set_xlabel("Volume (ml)")
            ax1.set_ylabel("Voltage (mV)", color=color)
            ax1.plot(x, y, color=color, antialiased=True, label="Voltage")
            ax1.tick_params(axis="y", labelcolor=color)
            if len(y) > 3:
                try:
                    popt, _ = curve_fit(self.poly_func, x, y, p0=[max(y) * 3 / 4, -max(x), max(y), 1.5])
                    print(f"电位突跃点：{-popt[1] / popt[3]:.3f}")
                    x_d = np.arange(0, max(x), 0.05)
                    y_fit = self.poly_func(x_d, *popt)
                    dE_dV = np.gradient(y_fit)
                    d2E_dV2 = np.gradient(dE_dV)
                    ax3 = ax1.twinx()
                    color3 = "tab:green"
                    ax3.set_ylabel("2nd Derivative", color=color3)
                    ax3.plot(x_d, d2E_dV2, color=color3)
                    ax3.tick_params(axis="y", labelcolor=color3)
                    ax3.grid(True, linestyle="--", linewidth=0.5, color="gray", axis="both")
                except Exception as e:
                    print(f"曲线拟合失败: {e}")
        else:
            ax1.set_xlabel("Volume (ml)")
            ax1.set_ylabel("End_Point", color="tab:red")
        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("Color (0:not_end, 1:end_point)", color=color2)
        ax2.plot(x, z, color=color2, marker="o", markersize=3, label="Color")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Not Changed", "Changed"])
        ax2.spines["right"].set_position(("outward", 60 if has_voltage else 0))
        first_1_index = next((i for i, value in enumerate(z) if value == 1), None)
        if first_1_index is not None:
            x_first_1 = x[first_1_index]
            ax1.axvline(x=x_first_1, color="purple", linestyle="--", alpha=0.7, label="Color Change Point")
            ax1.annotate(
                f"Color Change: {x_first_1:.2f} ml",
                xy=(x_first_1, ax1.get_ylim()[1] if has_voltage else 1),
                xytext=(x_first_1 + 0.5, ax1.get_ylim()[1] * 0.8 if has_voltage else 0.8),
                arrowprops=dict(arrowstyle="->", color="purple"),
            )
        fig.tight_layout()
        if not os.path.exists("Output"):
            os.makedirs("Output")
        plt.savefig(f"Output/{self.formatted_start_time}.png")
        plt.show()
        plt.pause(1)
        plt.close()

    def predictor(self, im_file):
        image = Image.open(im_file)
        data_transform = transforms.Compose([
            transforms.Resize(MATConfig.RESIZE_SIZE),
            transforms.CenterCrop(MATConfig.CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MATConfig.IMAGENET_MEAN, MATConfig.IMAGENET_STD),
        ])
        img = data_transform(image)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        class_a = self.class_indict[str(predict_cla)]
        prob_b = round(float(predict[predict_cla].numpy()), 3)
        self.last_class = class_a
        self.last_prob = prob_b
        print("class_:", class_a)
        print("prob_:", prob_b)
        return class_a, prob_b

    def predict_with_smoothing(self, im_file, verbose=True):
        n_classes = len(self.class_indict)
        with torch.no_grad():
            image = Image.open(im_file)
            data_transform = transforms.Compose([
                transforms.Resize(MATConfig.RESIZE_SIZE),
                transforms.CenterCrop(MATConfig.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MATConfig.IMAGENET_MEAN, MATConfig.IMAGENET_STD),
            ])
            img = torch.unsqueeze(data_transform(image), dim=0)
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            probs = torch.softmax(output, dim=0).tolist()
        predict_cla = max(range(n_classes), key=lambda i: probs[i])
        class_a = self.class_indict[str(predict_cla)]
        prob_b = round(probs[predict_cla], 3)
        self.last_class = class_a
        self.last_prob = prob_b
        self.prob_buffer.append(probs)
        if len(self.prob_buffer) > self.smooth_window:
            self.prob_buffer.pop(0)
        avg_probs = [
            sum(p[i] for p in self.prob_buffer) / len(self.prob_buffer)
            for i in range(n_classes)
        ]
        label_names = [self.class_indict[str(i)] for i in range(n_classes)]
        prob_map = dict(zip(label_names, avg_probs))
        p_red = prob_map.get("wine_red", 0.0)
        p_purple = prob_map.get("purple", 0.0)
        p_blue = prob_map.get("blue", 0.0)
        if verbose:
            print(f"  raw: {class_a} ({prob_b:.3f})  smoothed: red={p_red:.3f} purple={p_purple:.3f} blue={p_blue:.3f} | state={self.state}")
        if self.state == "FAST":
            if p_blue > self.blue_threshold:
                self.consecutive_blue += 1
                if self.consecutive_blue == 1 and self.first_blue_volume is None:
                    self.first_blue_volume = self.total_volume
            else:
                self.consecutive_blue = 0
            if self.transition_class:
                if p_purple > self.purple_threshold:
                    self.consecutive_purple += 1
                else:
                    self.consecutive_purple = 0
            if self.consecutive_blue >= self.confirm_frames:
                if verbose:
                    print(f"\n----->> [FAST→STOP] blue持续{self.confirm_frames}帧, 终点! <<-----")
                self.state = "STOP"
                self.transition_detected = True
            elif self.transition_class and self.consecutive_purple >= self.confirm_frames:
                if verbose:
                    print(f"\n----->> [FAST→SLOW] purple持续{self.confirm_frames}帧, 减速! <<-----")
                self.state = "SLOW"
                self.transition_detected = True
            elif not self.transition_class and self.consecutive_blue >= 1:
                if verbose:
                    print(f"\n----->> [FAST→SLOW] 首次蓝色, 减速! <<-----")
                self.state = "SLOW"
                self.transition_detected = True
        elif self.state == "SLOW":
            self.slow_loop_count += 1
            blue_active = p_blue > self.blue_threshold or (p_blue - p_purple > 0.15 and p_blue > 0.30)
            if blue_active:
                self.consecutive_blue += 1
                if self.consecutive_blue == 1 and self.first_blue_volume is None:
                    self.first_blue_volume = self.total_volume
            else:
                self.consecutive_blue = 0
            if self.consecutive_blue >= self.confirm_frames:
                if verbose:
                    print(f"\n----->> [SLOW→STOP] blue持续{self.confirm_frames}帧, 终点! <<-----")
                self.state = "STOP"
            elif self.slow_loop_count >= 25:
                if verbose:
                    print(f"\n----->> [SLOW→STOP] 慢速超时({self.slow_loop_count}轮), 强制终点! <<-----")
                self.state = "STOP"
        if verbose:
            decision = {"FAST": "快速滴定", "SLOW": "缓慢滴定", "STOP": "停止"}[self.state]
            print(f"  decision: {decision}")
        return class_a, prob_b, self.state

    def save_feedback(self, frame, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_dir = os.path.join(self.feedback_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        filename = f"feedback_{timestamp}_{self.total_volume:.3f}ml.jpg"
        filepath = os.path.join(label_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"\n  >>> 反馈已保存: {label} -> {filepath}")

    def apply_white_balance(self, frame):
        try:
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - MATConfig.WB_TARGET) * (result[:, :, 0] / 255.0) * MATConfig.WB_FACTOR)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - MATConfig.WB_TARGET) * (result[:, :, 0] / 255.0) * MATConfig.WB_FACTOR)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    def run(self, quick_speed=0.2, slow_speed=0.05, switching_point=5, end_kind="blue", end_prob=0.7,
            camera_delay=0.0, transition_speed=None,
            smooth_window=3, blue_threshold=0.40, purple_threshold=0.45, confirm_frames=2):
        total_n = self.overdose
        first_dispense = True
        if transition_speed is None:
            transition_speed = slow_speed
        self.smooth_window = smooth_window
        self.blue_threshold = blue_threshold
        self.purple_threshold = purple_threshold
        self.confirm_frames = confirm_frames
        self.state = "FAST"
        self.prob_buffer = []
        self.consecutive_blue = 0
        self.consecutive_purple = 0
        self.transition_detected = False
        self.first_blue_volume = None
        self.slow_loop_count = 0
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: 不可用 (使用CPU)")
        self.start_time = time.time()
        self.formatted_start_time = datetime.fromtimestamp(self.start_time).strftime("%Y%m%d_%H%M%S")
        print(f"滴定开始时间: {format_date_time(self.formatted_start_time)}")
        print(f"状态机参数: window={smooth_window}, blue_thr={blue_threshold}, purple_thr={purple_threshold}, confirm={confirm_frames}")
        while True:
            print("=" * 50)
            if self.now_volume <= 0:
                if not self.start_move_1():
                    print("错误: 抽取失败")
                    break
                self.now_volume += MATConfig.DRAW_VOLUME
            if first_dispense:
                print("开始加液...")
                first_dispense = False
            if self.transition_detected:
                speed = slow_speed
            elif self.transition_class:
                speed = quick_speed
            else:
                speed = quick_speed if self.total_volume < switching_point else slow_speed
            if not self.start_move_2(speed):
                print("错误: 移动失败，停止实验")
                break
            self.total_volume += speed * self.volume_par
            self.now_volume -= speed
            self.total_volume = round(self.total_volume, MATConfig.VOLUME_DECIMALS)
            self.volume_list.append(self.total_volume)
            print(f"Current Total Volume: {self.total_volume} ml")
            if self.now_volume <= 0:
                if not self.start_move_1():
                    print("错误: 抽取失败")
                    break
                self.now_volume += MATConfig.DRAW_VOLUME
                print(f"已补液 {MATConfig.DRAW_VOLUME}ml, 剩余 {self.now_volume:.1f}ml")
            if camera_delay:
                time.sleep(camera_delay)
            if self.type in ("Vision", "All"):
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from camera.")
                    break
                name = self.get_picture(frame, self.formatted_start_time)
                im_file = os.path.join("Input", name)
                class_a, prob_b, state = self.predict_with_smoothing(im_file)
                state_label = {"FAST": "快速滴定", "SLOW": "缓慢滴定", "STOP": "已停止"}[state]
                display = frame.copy()
                cv2.putText(display, f"State: {state_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Pred: {class_a} ({prob_b:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, "F1=Red F2=Purple F3=Blue", (10, display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("MAT Titration", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("1") or key == 0x70:
                    self.save_feedback(frame, "wine_red")
                elif key == ord("2") or key == 0x71:
                    self.save_feedback(frame, "purple")
                elif key == ord("3") or key == 0x72:
                    self.save_feedback(frame, "blue")
                elif key == ord("q"):
                    print("\n用户退出")
                    break
                if state == "STOP":
                    if total_n == self.overdose:
                        print("----->>视觉终点检测到!<<-----")
                        endpoint_vol = self.first_blue_volume if self.first_blue_volume is not None else self.total_volume
                        print(f"Total Volume: {endpoint_vol} ml")
                        print(f"Image File: {im_file}")
                        self.finish_volume = endpoint_vol
                        self.finish_picture = im_file
                        shutil.copy(im_file, f"Output/final_{self.formatted_start_time}.jpg")
                        print(f"终点图片已备份到: Output/final_{self.formatted_start_time}.jpg")
                    self.color_list.append(1)
                    total_n -= 1
                elif state == "SLOW":
                    if not self.transition_detected:
                        print(f"\n----->>过渡色(purple)检测到，减速！<<-----")
                        print(f"Total Volume: {self.total_volume} ml")
                        self.transition_detected = True
                    self.color_list.append(0.5)
                else:
                    self.color_list.append(0)
            if self.type in ("Potential", "All"):
                try:
                    voltage = self.voltage()
                    self.voltage_list.append(voltage)
                    print(f"Current Voltage: {voltage} mV")
                except Exception:
                    self.voltage_list.append(0)
            if total_n < 0:
                print("达到过量滴定次数，实验结束")
                break
            if self.total_volume >= self.final_volume:
                print("达到目标体积，实验结束")
                break
        if not self.finish_picture and self.volume_list:
            print("警告: 未检测到视觉终点！")
            input_dir = os.path.join(self.data_root, "Input")
            if os.path.exists(input_dir):
                images = [f for f in os.listdir(input_dir) if f.endswith(".jpg") and self.formatted_start_time in f]
                if images:
                    images.sort()
                    last_image = images[-1]
                    self.finish_picture = os.path.join("Input", last_image)
                    self.finish_volume = self.total_volume
                    print(f"使用最后一张图片作为终点: {self.finish_picture}")
        return self.save_results()

    def save_results(self):
        end_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        print("=" * 50)
        print("实验完成！")
        print(f"滴定开始时间: {format_date_time(self.formatted_start_time)}")
        print(f"实验结束时间: {format_date_time(end_time)}")
        print(f"最终滴定体积: {self.finish_volume} ml")
        print(f"Volume List: {self.volume_list}")
        print(f"Voltage List: {self.voltage_list}")
        print(f"Color List: {self.color_list}")
        print("=" * 50)
        final_data = {
            "start_time": format_date_time(self.formatted_start_time),
            "end_time": format_date_time(end_time),
            "volume_record": json.dumps(self.volume_list),
            "voltage_record": json.dumps(self.voltage_list),
            "color_record": json.dumps(self.color_list),
            "final_volume": str(self.finish_volume),
        }
        if not os.path.exists("Output"):
            os.makedirs("Output")
        output_file = f"Output/{self.formatted_start_time}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "start_time": format_date_time(self.formatted_start_time),
                    "end_time": format_date_time(end_time),
                    "volume_record": self.volume_list,
                    "voltage_record": self.voltage_list,
                    "color_record": self.color_list,
                    "final_volume": self.finish_volume,
                    "final_image": self.finish_picture,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"数据已保存到: {output_file}")
        return final_data, self.finish_picture

    def close_serial(self):
        try:
            if hasattr(self, "pump_ser") and self.pump_ser is not None:
                if self.pump_ser.is_open:
                    self.pump_ser.close()
                self.pump_ser = None
                print("串口已关闭")
        except Exception as e:
            print(f"关闭串口时出错: {e}")

    def close_camera(self):
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
                self.cap = None
                print("摄像头已关闭")
        except Exception as e:
            print(f"关闭摄像头时出错: {e}")


def run_titration_standalone():
    setup_environment()
    CAMERA_INDEX = 1
    MODEL_NAME = "EBT"
    N_CLASSES = 2
    TRANSITION_CLASS = None
    OVERDOSE_COUNT = 0
    VOLUME_PAR = 0.913
    FINAL_VOLUME = 100
    TYPE = "Vision"
    QUICK_SPEED = 0.3
    SLOW_SPEED = 0.08
    TRANSITION_SPEED = 0.25
    BLUE_THRESHOLD = 0.40
    SMOOTH_WINDOW = 3
    CONFIRM_FRAMES = 2
    print("初始化MAT实例...")
    mat = MAT(
        video_source_index=CAMERA_INDEX,
        model_name=MODEL_NAME,
        classes=N_CLASSES,
        overdose=OVERDOSE_COUNT,
        volume_par=VOLUME_PAR,
        final_volume=FINAL_VOLUME,
        typ=TYPE,
        transition_class=TRANSITION_CLASS,
    )
    print("开始运行滴定实验...")
    final_data, finish_picture = mat.run(
        quick_speed=QUICK_SPEED,
        slow_speed=SLOW_SPEED,
        transition_speed=TRANSITION_SPEED,
        blue_threshold=BLUE_THRESHOLD,
        smooth_window=SMOOTH_WINDOW,
        confirm_frames=CONFIRM_FRAMES,
    )
    print("\n滴定完成！")
    print(f"终点体积: {final_data['final_volume']} ml")
    if os.path.exists("Output"):
        charts = [f for f in os.listdir("Output") if f.endswith(".png")]
        if charts:
            print(f"滴定曲线已保存: Output/{charts[-1]}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("Mlabs AI Titration System v1.5")
    print("基于计算机视觉的自动滴定实验系统")
    print("=" * 60)
    print("\n运行模式:")
    print("  1) python titration.py          独立模式 (无需登录)")
    print("  2) python titration.py --demo    演示模式 (模拟滴定, 无需硬件)")
    print("  3) python titration.py --upload  完整模式 (含平台登录与上传)")
    print()
    import sys
    if "--demo" in sys.argv:
        print("演示模式 - 模拟滴定流程...")
        run_titration_standalone()
    elif "--upload" in sys.argv:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
        root = tk.Tk()
        root.withdraw()
        username = simpledialog.askstring("登录", "用户名:")
        password = simpledialog.askstring("登录", "密码:", show="*")
        if username and password:
            token = login_to_platform(username, password)
            if token:
                try:
                    run_titration_standalone()
                except Exception as e:
                    print(f"程序运行出错: {e}")
                    traceback.print_exc()
            else:
                print("登录失败")
        else:
            print("取消登录")
    else:
        run_titration_standalone()
