"""串口自动检测工具 - 优先 CH340，其次 USB 串行，最后任意串口"""

import serial.tools.list_ports


def _list_ch340_ports():
    ports = serial.tools.list_ports.comports()
    ch340_ports = []
    for port in ports:
        if "CH340" in port.description or "CH340" in port.device:
            ch340_ports.append(port.device)
            print("Found CH340 ports:", port.device)
    return ch340_ports


def _list_usb_ports():
    ports = serial.tools.list_ports.comports()
    usb_ports = []
    for port in ports:
        if "串行" in port.description or "串行" in port.device:
            usb_ports.append(port.device)
            print("Found USB ports:", port.device)
    return usb_ports


def find_any_port():
    """查找可用串口，依次尝试: CH340 > USB串行 > 任意串口"""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    for p in ports:
        if "CH340" in p.description or "CH340" in p.device:
            print(f"Found CH340 port: {p.device}")
            return p.device

    for p in ports:
        if "串行" in p.description or "串行" in p.device:
            print(f"Found USB port: {p.device}")
            return p.device

    for p in ports:
        print(f"Fallback to port: {p.device}")
        return p.device

    return None


if __name__ == "__main__":
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No port available")
    else:
        for port in ports:
            print(port)

    found = find_any_port()
    if found:
        print(f"自动检测到串口: {found}")
    else:
        print("未检测到串口设备")
