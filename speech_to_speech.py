import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
from TTS.api import TTS
import torch
from ollama import Client
import warnings
import time
import wave
import keyboard
from datetime import datetime
warnings.filterwarnings("ignore")

class GrabadorAudio:
    def __init__(self, frecuencia_muestreo=16000):
        self.frecuencia = frecuencia_muestreo
        self.frames = []
        self.grabando = False
        
    def callback(self, indata, frames, time, status):
        if self.grabando:
            self.frames.append(indata.copy())
    
    def iniciar_grabacion(self):
        self.frames = []
        self.grabando = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.frecuencia,
            callback=self.callback
        )
        self.stream.start()
        
    def detener_grabacion(self):
        self.grabando = False
        self.stream.stop()
        self.stream.close()
        
    def guardar_audio(self, ruta_archivo):
        if not self.frames:
            return False
            
        audio_data = np.concatenate(self.frames, axis=0)
        sf.write(ruta_archivo, audio_data, self.frecuencia)
        return True

class ChatPorVoz:
    def __init__(self, modelo_whisper="large-v3", usar_gpu=True):
        # Inicializar Whisper para voz a texto
        tipo_computo = "float16" if usar_gpu else "int8"
        dispositivo = "cuda" if usar_gpu else "cpu"
        self.whisper = WhisperModel(modelo_whisper, device=dispositivo, compute_type=tipo_computo)
        
        # Inicializar TTS para texto a voz
        self.tts = TTS("tts_models/es/css10/vits", gpu=usar_gpu)
        
        # Inicializar cliente Ollama
        self.ollama = Client(host='http://localhost:11434')
        
        # Inicializar grabador
        self.grabador = GrabadorAudio()
        
        # Crear directorio para archivos temporales
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def transcribir_audio(self, ruta_audio):
        """Convertir voz a texto usando Whisper"""
        segmentos, _ = self.whisper.transcribe(ruta_audio, beam_size=5, language="es")
        return " ".join([segmento.text for segmento in segmentos])
    
    def procesar_con_llm(self, texto):
        """Procesar texto con LLM local usando Ollama"""
        respuesta = self.ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': 'Eres un asistente conversacional amigable. Responde de manera natural y concisa en español.'
            },
            {
                'role': 'user',
                'content': texto
            }
        ])
        return respuesta['message']['content']
    
    def generar_voz(self, texto, ruta_salida):
        """Convertir texto a voz usando TTS"""
        self.tts.tts_to_file(text=texto, file_path=ruta_salida)
    
    def reproducir_audio(self, ruta_archivo):
        """Reproducir archivo de audio"""
        data, fs = sf.read(ruta_archivo)
        sd.play(data, fs)
        sd.wait()
    
    def iniciar_chat(self):
        print("\n=== Chat por Voz Iniciado ===")
        print("Mantén presionada la tecla 'R' mientras hablas")
        print("Suelta 'R' para procesar tu mensaje")
        print("Di 'salir' para terminar la conversación")
        print("================================\n")
        
        while True:
            # Esperar a que el usuario presione 'R'
            print("\nMantén presionada 'R' para hablar...")
            keyboard.wait('r')
            
            # Iniciar grabación
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo_entrada = os.path.join(self.temp_dir, f"entrada_{timestamp}.wav")
            archivo_salida = os.path.join(self.temp_dir, f"salida_{timestamp}.wav")
            
            print("Grabando... (suelta 'R' para terminar)")
            self.grabador.iniciar_grabacion()
            
            # Esperar a que el usuario suelte 'R'
            keyboard.wait('r', suppress=True, trigger_on_release=True)
            
            # Detener grabación
            print("Procesando...")
            self.grabador.detener_grabacion()
            if not self.grabador.guardar_audio(archivo_entrada):
                print("No se detectó audio. Intenta de nuevo.")
                continue
            
            # Procesar audio
            texto = self.transcribir_audio(archivo_entrada)
            print(f"\nTú: {texto}")
            
            # Verificar si el usuario quiere salir
            if "salir" in texto.lower():
                print("\n¡Hasta luego! Terminando chat...")
                break
            
            # Obtener respuesta del LLM
            respuesta = self.procesar_con_llm(texto)
            print(f"Asistente: {respuesta}")
            
            # Generar y reproducir respuesta
            self.generar_voz(respuesta, archivo_salida)
            print("Reproduciendo respuesta...")
            self.reproducir_audio(archivo_salida)

def main():
    print("Iniciando sistema de chat por voz...")
    print("Verificando CUDA:", "Disponible" if torch.cuda.is_available() else "No disponible")
    
    chat = ChatPorVoz(
        modelo_whisper="large-v3",
        usar_gpu=torch.cuda.is_available()
    )
    
    try:
        chat.iniciar_chat()
    except KeyboardInterrupt:
        print("\nChat terminado por el usuario.")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nLimpiando archivos temporales...")
        # Opcionalmente, limpiar archivos temporales aquí

if __name__ == "__main__":
    main()