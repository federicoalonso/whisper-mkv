import torch
import whisper

# Verificar CUDA
print("CUDA disponible:", torch.cuda.is_available())
print("Versión de PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("Dispositivo CUDA:", torch.cuda.get_device_name(0))
    print("Número de GPUs:", torch.cuda.device_count())
    
    # Intentar crear un tensor en GPU
    x = torch.tensor([1, 2, 3]).cuda()
    print("Tensor creado en:", x.device)
    
    # Cargar modelo pequeño de whisper para prueba
    model = whisper.load_model("tiny").cuda()
    print("Modelo cargado en:", next(model.parameters()).device)