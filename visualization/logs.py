import time

def log(msg, agent=None):
    prefix = f"[{agent}]" if agent else "[SYSTEM]"
    print(f"{prefix} {time.strftime('%H:%M:%S')} | {msg}")