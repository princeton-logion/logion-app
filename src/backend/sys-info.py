import platform
import psutil
import cpuinfo
import subprocess
import GPUtil
import json



def check_apple_m_chip():
    """
    Detects if the machine has an Apple M-series chip.
    """
    darwin_version = float(platform.release().split('.')[0])
    
    if darwin_version and darwin_version >= 20:  # Darwin 20 first w/ m-chip
        try:
            output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if "Apple" in output:
                return output
        except Exception:
            return f"Unable to detect Apple GPU."
    return None



def fetch_gpu():
    """
    Detect GPU info, including M-series integrated GPUs on macOS.
    """
    gpus = []
    
    if platform.system() == "Darwin":
        m_chip_info = check_apple_m_chip()
        if m_chip_info is not None:
            gpus.append(m_chip_info)
        try:
            output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
            for line in output.split("\n"):
                if "Chip:" in line or "VRAM" in line:
                    gpus.append(line.strip().replace("Chip: ", "").replace("VRAM (Total): ", ""))
        except Exception:
            gpus.append(f"Unable to detect Apple GPU.")
    elif platform.system() == "Windows":
        try:
            output = subprocess.check_output(["dxdiag", "/t", "dxdiag_output.txt"], text=True)
            with open("dxdiag_output.txt", "r", encoding="utf-8") as f:
                dxdiag_data = f.read()
            for line in dxdiag_data.split("\n"):
                if "Card name:" in line:
                    gpus.append(line.split("Card name:")[-1].strip())
        except Exception:
            gpus.append("Unable to detect Windows GPU.")
    else:
        try:
            detected_gpus = GPUtil.getGPUs()
            for gpu in detected_gpus:
                gpus.append(f"{gpu.name} ({gpu.memoryTotal} GB)")
        except Exception:
            gpus.append("Unable to detect Nvidia GPU.")

    return gpus



def fetch_sys_info():
    system_info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": str(cpuinfo.get_cpu_info().get('brand_raw', "Not Found")),
        "cores": str(psutil.cpu_count(logical=True)),
        "gpu": fetch_gpu(),
        "ram": str(round(psutil.virtual_memory().total / (1024 ** 3))) # convert B to GB
    }
    print(system_info)
    return system_info



if __name__ == "__main__":
    fetch_sys_info()
