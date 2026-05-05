import time
import socket
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--r", required=True)
args = parser.parse_args()

print("✅ Python script started")
print(f"📍 Running on host: {socket.gethostname()}")
print(f"👤 User: {os.getenv('USER')}")
print(f"📥 Argument received: {args.r}")

# Simple numpy test
arr = np.array([1, 2, 3, 4, 5])
arr_sum = np.sum(arr)
arr_mean = np.mean(arr)

print("✅ numpy imported successfully")
print(f"🔢 Array: {arr}")
print(f"➕ Sum: {arr_sum}")
print(f"📊 Mean: {arr_mean}")

# Create result folder
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Write output file
result_file = os.path.join(result_dir, "output.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("This is a test output file.\n")
    f.write(f"Host: {socket.gethostname()}\n")
    f.write(f"User: {os.getenv('USER')}\n")
    f.write(f"Argument received: {args.r}\n")
    f.write(f"Numpy array: {arr}\n")
    f.write(f"Sum: {arr_sum}\n")
    f.write(f"Mean: {arr_mean}\n")

print(f"📁 Result folder created: {result_dir}")
print(f"📝 File written: {result_file}")

for i in range(5):
    print(f"⏱️ Step {i+1}/5 running...")
    time.sleep(1)

print("🎉 Script finished successfully")