#!/usr/bin/env python3
"""Read ESP32-S3 USB Serial/JTAG console.

The device only emits console output while the host asserts DTR (it treats a
deasserted DTR as "no terminal attached" and drops the bytes), so DTR must be
held high for the whole session.
"""
import sys, time, serial

port = sys.argv[1] if len(sys.argv) > 1 else '/dev/cu.usbmodem5C630570441'
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
reset = '--reset' in sys.argv

s = serial.Serial(port, 115200, timeout=0.3)
if reset:
    s.setDTR(False); s.setRTS(True); time.sleep(0.15)
    s.setRTS(False); time.sleep(0.05)
s.setDTR(True); s.setRTS(False)          # terminal attached
t0 = time.time()
while time.time() - t0 < secs:
    line = s.readline()
    if line:
        sys.stdout.write(line.decode('utf-8', 'replace'))
        sys.stdout.flush()
s.close()
