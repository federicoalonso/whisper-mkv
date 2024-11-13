import torch
from TTS.api import TTS
import sys

def test_cuda():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    
    try:
        # Intenta cargar un modelo TTS simple
        tts = TTS("tts_models/es/css10/vits", gpu=True)
        print("TTS cargado correctamente con GPU")
    except Exception as e:
        print("Error al cargar TTS:", str(e))

if __name__ == "__main__":
    test_cuda()