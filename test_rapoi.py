import time
import socket
import os

print("✅ Python script started")
print(f"📍 Running on host: {socket.gethostname()}")
print(f"👤 User: {os.getenv('USER')}")

for i in range(10):
    print(f"⏱️ Step {i+1}/5 running...")
    time.sleep(1)

print("🎉 Script finished successfully")
