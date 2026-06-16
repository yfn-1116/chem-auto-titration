"""注射泵完整通信诊断 v2 — 模拟主程序协议，排查卡住原因"""

import serial
import serial.tools.list_ports
import time
import re
import Find_COM

BAUDRATE = 115200
TIMEOUT = 0.5
MOVE_TIMEOUT = 30

print("=" * 60)
print("  注射泵通信诊断 v2 - 模拟主程序协议")
print("=" * 60)

# 1. 找串口
found = Find_COM.find_any_port()
if not found:
    print("未找到串口")
    exit()
print(f"串口: {found}")

# 2. 打开
ser = serial.Serial(found, BAUDRATE, timeout=2)
print(f"已打开 {found} @ {BAUDRATE}")

def get_pump_status():
    try:
        ser.reset_input_buffer()
        time.sleep(0.05)
        response = ser.read_all().decode("utf-8", errors="replace")
        if not response:
            return None
        matches = re.findall(r"S\|([0-9A-Fa-f]+)", response, re.IGNORECASE)
        if not matches:
            print(f"  无S|包, 原始: {response[:80]}")
            return None
        last_match = matches[-1]
        packed_value = int(last_match, 16)
        return {
            "v_int": (packed_value >> 45) & 0x1FFFF,
            "b_int": (packed_value >> 44) & 0x1,
            "h_int": (packed_value >> 43) & 0x1,
            "p_int": (packed_value >> 27) & 0xFFFF,
            "volume_ml": ((packed_value >> 45) & 0x1FFFF) / 1000.0,
            "is_busy": bool((packed_value >> 44) & 0x1),
            "is_homing": bool((packed_value >> 43) & 0x1),
            "raw_hex": last_match,
            "raw_int": packed_value,
        }
    except Exception as e:
        print(f"  解析异常: {e}")
        return None

def send_cmd(cmd, desc=""):
    print(f"\n>>> 发送: {cmd!r} {desc}")
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(cmd.encode() if isinstance(cmd, str) else cmd)
    time.sleep(1)
    resp = ser.read_all()
    text = resp.decode("utf-8", errors="replace")
    print(f"  响应({len(resp)}B): {text[:200]}")
    if "ERROR" in text:
        print(f"  ⚠ 错误: {text[:100]}")
    status = get_pump_status()
    if status:
        print(f"  状态: vol={status['volume_ml']:.3f}ml  busy={status['is_busy']}  homing={status['is_homing']}  hex={status['raw_hex']}")
    return status

def wait_for_move_complete(label="", timeout=MOVE_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        status = get_pump_status()
        if status is None:
            time.sleep(0.2)
            continue
        if not status["is_busy"]:
            print(f"  ✅ [{label}] 完成! vol={status['volume_ml']:.3f}ml  hex={status['raw_hex']}")
            return True
        print(f"  ⏳ [{label}] 等待中... busy=1  hex={status['raw_hex']}")
        time.sleep(0.2)
    print(f"  ❌ [{label}] 超时!")
    return False

# 3. 测试归零
print("\n" + "="*60)
print("阶段1: 归零 RE")
print("="*60)
ser.reset_input_buffer()
send_cmd("RE\n", "(归零)")
time.sleep(2)
# 持续读状态直到不忙
wait_for_move_complete("归零", timeout=15)

# 4. 测试抽取
print("\n" + "="*60)
print("阶段2: 抽取 IP|20,CCW")
print("="*60)
ser.reset_input_buffer()
status = send_cmd("IP|20,CCW\n", "(抽取20ml)")
if status and not status["is_busy"]:
    wait_for_move_complete("抽取", timeout=15)

# 5. 测试加液
print("\n" + "="*60)
print("阶段3: 加液 IP|0.1,CW")
print("="*60)
ser.reset_input_buffer()
status = send_cmd("IP|0.1,CW\n", "(加液0.1ml)")
if status and not status["is_busy"]:
    wait_for_move_complete("加液", timeout=15)

# 6. 诊断状态位
print("\n" + "="*60)
print("阶段4: 状态位诊断")
print("="*60)
ser.reset_input_buffer()
ser.write(b"RE\n")
time.sleep(1)
resp = ser.read_all()
text = resp.decode("utf-8", errors="replace")
matches = re.findall(r"S\|([0-9A-Fa-f]+)", text, re.IGNORECASE)
for i, m in enumerate(matches):
    val = int(m, 16)
    vol = (val >> 45) & 0x1FFFF
    busy = (val >> 44) & 0x1
    home = (val >> 43) & 0x1
    volt = (val >> 27) & 0xFFFF
    print(f"  S|{m}")
    print(f"    int={val}")
    print(f"    hex_bin={val:064b}")
    print(f"    vol_ml={vol/1000.0:.3f}  busy={busy}  homing={home}  voltage_mV={volt}")
    print(f"    ------- 反向检查 -------")
    print(f"    bit63:{(val>>63)&1}  bit44:{busy}  bit43:{home}  bit27-42:{volt}")

ser.close()
print(f"\n串口已关闭")

print("\n" + "="*60)
print("诊断结论:")
print("="*60)
print("1. 如果归零成功但抽取失败 → 检查 IP|20,CCW 命令格式是否被泵支持")
print("2. 如果所有命令都返回 Injector busy → 泵卡在某个状态,需手动归零或重启")
print("3. 如果状态解析全是0 → 位域定义 (>>45, >>44 等) 与该泵的实际输出不匹配")
print("4. 如果串口无响应 → 检查波特率或接线")
