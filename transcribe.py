import whisper
import torch
import os
from datetime import datetime

def transcribe_audio(file_path, model_name="medium", language="es"):
    """
    Transcribe un archivo de audio a texto usando GPU si está disponible.
    
    Args:
        file_path (str): Ruta al archivo .mkv
        model_name (str): Nombre del modelo de Whisper a usar
        language (str): Idioma del audio (es para español)
        
    Returns:
        str: Texto transcrito
    """
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    # Información del sistema
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== Información del Sistema ===")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Memoria GPU Disponible: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB usado")
    
    # Cargar el modelo
    print(f"\nCargando modelo {model_name}...")
    start_time = datetime.now()
    model = whisper.load_model(model_name).to(device)
    print(f"Modelo cargado en {(datetime.now() - start_time).total_seconds():.2f} segundos")
    
    # Realizar la transcripción
    print("\nIniciando transcripción...")
    start_time = datetime.now()
    try:
        result = model.transcribe(
            file_path,
            language=language,
            verbose=True
        )
        
        tiempo_transcripcion = (datetime.now() - start_time).total_seconds()
        print(f"\nTranscripción completada en {tiempo_transcripcion:.2f} segundos")
        
        # Liberar memoria GPU si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Memoria GPU liberada")
        
        return result["text"]
        
    except Exception as e:
        print(f"Error durante la transcripción: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

if __name__ == "__main__":
    # Configuración
    archivo_mkv = "video/place_to_pay.mkv"
    
    try:
        print(f"Procesando archivo: {archivo_mkv}")
        
        # Realizar la transcripción
        texto = transcribe_audio(archivo_mkv)
        
        # Guardar el resultado en un archivo
        nombre_salida = archivo_mkv.rsplit(".", 1)[0] + "_transcripcion.txt"
        with open(nombre_salida, "w", encoding="utf-8") as f:
            f.write(texto)
            
        print(f"\nTranscripción guardada en: {nombre_salida}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        
    finally:
        # Asegurarse de liberar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()