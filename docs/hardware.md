# περῐ́ τῆς μηχᾰνῆς

Hardware affects software. Logion does not require access to high-performance computing (HPC) resources. It is designed to run on local machines with diverse hardware and operating systems. Hardware does not affect the accuracy or quality of Logion's predictions. But hardware *does* affect the speed at which Logion generates predictions. As such, for perhaps the first time in history, philologists may want to be aware of their computer's hardware. Here is a brief guide to help users navigate hardware questions in relation to Logion.

## CPU vs. GPU

Nearly every computer nowadays has both a central processing unit (CPU) and graphics processing unit (GPU). Both are processors that use memory to store/access data, perform calculations, and execute instructions. CPUs handle general-purpose computing tasks (word processors, web browsing, etc.); GPUs handle computationally expensive, high-throughput tasks (video games, training AI models, etc.).

## How hardware affects Logion?

Logion is designed to run on both a local machine's CPU and GPU. When a user starts Logion, the software detects which type of GPU is available on the host machine. If the machine has a compatible GPU, it utilizes that GPU to generate any predictions. If a GPU is unavailable, Logion defaults to the machine's CPU. Logion is compatible with the following types of GPUs:

- Nvidia
- Apple Silicon (M-chip)
- Intel Graphics

Our team's internal experiments suggest that, when executing Logion's error detection task on a sequence of 122 words in Byzantine Greek, CPUs on both Mac and Windows devices require about 20 minutes of processing time. By comparison, an Apple M-chip requires about 8 minutes of processing time.

## How to find your computer's GPU

#### On macOS
1. Go to the top-left corner of the computer menu bar.
2. Click the Apple Logo ().
3. Select **About This Mac** from the drop-down menu.
4. GPU information appears beside **Graphics** or **Chip**.

#### On Windows
1. Right-click on the taskbar.
2. Select **Task Manager**.
3. Click the **Performance** tab in the Task Manager .
4. Scroll down and select **GPU 0**.
5. GPU information appears in the top-right corner.
