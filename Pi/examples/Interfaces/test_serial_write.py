#!/usr/bin/env python3
"""Test script to verify serial communication with Arduino"""

import serial
import time

print("Testing serial communication...")

try:
    # Try to open the serial port
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    print(f"✓ Serial port opened: {ser.port}")
    print(f"  Baud rate: {ser.baudrate}")
    print(f"  Timeout: {ser.timeout}")
    
    time.sleep(2)
    print("\nClearing buffers...")
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    print("Sending 'R\\n' command...")
    ser.write(b'R\n')
    ser.flush()
    print("✓ Data sent to Arduino")
    print("  (Check if Arduino RX LED blinks)")
    
    print("\nWaiting for response (5 seconds)...")
    response = ser.readline()
    
    if response:
        print(f"✓ Received: {response}")
    else:
        print("✗ No response from Arduino (timeout)")
    
    ser.close()
    print("✓ Serial port closed")
    
except serial.SerialException as e:
    print(f"✗ Serial error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
