import sys
import platform

def check_system():
    print(f"Python version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print("Docker test successful!")

if __name__ == "__main__":
    check_system()
