"""注射泵串口通信诊断 — 只读不驱动，排查无法抽液的问题"""

import serial
import serial.tools.list_ports
import time
import re
import sys

print("=" * 60)
print("  注射泵通信诊断")
print("=" * 60)

# 1. 扫描可用串口
print("\n>>> 1. 扫描可用串口")
ports = list(serial.tools.list_ports.comports())
if not ports:
    print("  未找到任何串口！检查 USB 是否连接、驱动是否安装")
    sys.exit(1)

for p in ports:
    match_ch340 = "CH340" in p.description or "CH340" in p.device
    match_usb = "串行" in p.description or "串行" in p.device
    tag = ""
    if match_ch340:
        tag = " <-- CH340 优先匹配"
    elif match_usb:
        tag = " <-- USB串行匹配"
    print(f"  {p.device}: {p.description}{tag}")

# 2. 用 Find_COM 的逻辑找端口
import Find_COM
found = Find_COM.find_any_port()
if found:
    print(f"\n  Find_COM 结果: {found} ✅")
else:
    print(f"\n  Find_COM 结果: None ❌ (无法自动检测)")
    sys.exit(1)

# 3. 尝试打开串口
print(f"\n>>> 2. 尝试打开 {found} (波特率 115200)")
try:
    ser = serial.Serial(found, 115200, timeout=2)
    if ser.is_open:
        print(f"  打开成功 ✅ | 端口: {ser.port} | 波特率: {ser.baudrate}")
    else:
        print("  打开失败 ❌")
        sys.exit(1)
except Exception as e:
    print(f"  打开失败 ❌: {e}")
    print("  可能原因：端口被占用、权限不足、设备未就绪")
    sys.exit(1)

# 4. 发送归零命令并检查响应
print("\n>>> 3. 发送归零命令 RE")
try:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(b"RE\n")
    print(f"  已发送: RE")
    
    # 等待响应
    time.sleep(1)
    response = ser.read_all()
    print(f"  原始响应 ({len(response)} bytes): {response}")
    
    if response:
        try:
            text = response.decode("utf-8", errors="replace")
            print(f"  解码文本: {text[:200]}")
            matches = re.findall(r"S\|([0-9A-Fa-f]+)", text, re.IGNORECASE)
            if matches:
                print(f"  找到状态包: {matches[-1]}")
                val = int(matches[-1], 16)
                print(f"  解析值: 0x{val:X}")
                print(f"    volume: {(val >> 45) & 0x1FFFF}")
                print(f"    is_busy: {bool((val >> 44) & 0x1)}")
                print(f"    is_homing: {bool((val >> 43) & 0x1)}")
                print(f"    voltage: {(val >> 27) & 0xFFFF}")
            else:
                print(f"  未匹配到 S| 状态包格式 ❌")
                print(f"  提示: 检查协议格式是否正确")
        except Exception as e:
            print(f"  解码失败: {e}")
    else:
        print("  无响应 ❌")
        print("  可能原因：波特率不对、命令格式不对、泵未上电")
except Exception as e:
    print(f"  通信异常: {e}")

# 5. 发送抽取命令 (只测试发送，不实际执行以免出问题)
print("\n>>> 4. 测试抽取命令格式 (只测试发送，未执行)")
cmd = "IP|20,CCW\n"
print(f"  命令: {cmd!r}")
print(f"  编码: {cmd.encode('utf-8')}")

# 6. 发送加液命令格式测试
cmd2 = "IP|0.1,CW\n"
print(f"\n>>> 5. 测试加液命令格式")
print(f"  命令: {cmd2!r}")
print(f"  编码: {cmd2.encode('utf-8')}")

# 7. 读取泵状态（不发送命令）
print("\n>>> 6. 尝试读取泵状态 (发送 RE 后再读)")
try:
    ser.reset_input_buffer()
    ser.write(b"RE\n")
    time.sleep(2)
    resp = ser.read_all()
    if resp:
        print(f"  响应: {resp[:100]}")
    else:
        print("  无响应")
except Exception as e:
    print(f"  读取异常: {e}")

ser.close()
print(f"\n  串口已关闭")

print("\n" + "=" * 60)
print("  诊断完成")
print("=" * 60)
print()
print("常见问题排查:")
print("1. 没有任何串口 → 检查 USB 驱动 (CH340 需装驱动)")
print("2. 串口打开失败 → 检查是否被其他程序占用")
print("3. 发送命令无响应 → 检查波特率(115200)、命令格式")
print("4. 收到响应但无 S| 包 → 协议不匹配，需核对泵的通信协议")
print("5. 收到 S| 但解析异常 → 位域定义与泵的实际输出不符")
