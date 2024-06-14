(Instructions made on/for Kubuntu 24.04)

1. Creating virtual environment for the project:
python3 -m venv .

2. Switch to newly created venv:
source venv/bin/activate

3. Restoring dependencies:
pip install -r requirements.txt

4 (optional). To enable GPU acceleration for local models, it may be necessary to first install CUDA Toolkit and then running (keep in mind that the path for CUDACXX may change depending on the version):
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

5 (optional). If you have RTX 3090Ti or 3090 (or maybe even 3080Ti/3080), you may want to limit the clocks to avoid crashes (nvidia drivers for linux didn't seem to fix the stability issues):
sudo nvidia-smi -lgc 100,1700

6. To start the project, run:
streamlit run main.py
