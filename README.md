# Whisper MKV Transcriber

Una herramienta simple para transcribir archivos de audio MKV a texto usando el modelo Whisper de OpenAI.

## 🎯 Características

- Transcripción de archivos MKV a texto
- Soporte para GPU NVIDIA (CUDA)
- Múltiples modelos de Whisper disponibles (tiny, base, small, medium, large, turbo)
- Soporte multilenguaje
- Guardado automático de transcripciones

## 🛠️ Requisitos Previos

- Python 3.8 o superior
- NVIDIA GPU con soporte CUDA (recomendado)
- ffmpeg instalado en el sistema

### Instalación de ffmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows usando Chocolatey
choco install ffmpeg

# Windows usando Scoop
scoop install ffmpeg

# MacOS usando Homebrew
brew install ffmpeg
```

## ⚙️ Instalación

1. Crear un entorno virtual:
```bash
# Crear el entorno virtual
python -m venv env

# Activar el entorno virtual
# En Windows:
env\Scripts\activate
# En Linux/Mac:
source env/bin/activate
```

2. Instalar las dependencias:
```bash
# Instalar PyTorch con soporte CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar Whisper
pip install -U openai-whisper
```

## 📝 Uso

1. Coloca tus archivos MKV en la carpeta `video/`

2. Ejecuta el script:
```bash
python transcribe.py
```

El script generará un archivo de texto con el mismo nombre que el archivo MKV más el sufijo "_transcripcion.txt".

## 🔧 Configuración

Puedes modificar las siguientes variables en el script:

```python
model_name = "medium"  # Opciones: tiny, base, small, medium, large, turbo
language = "es"        # Código de idioma (es para español)
```

## 📊 Modelos Disponibles

| Modelo  | Parámetros | VRAM Requerida | Velocidad Relativa |
|---------|------------|----------------|-------------------|
| tiny    | 39 M      | ~1 GB          | ~10x             |
| base    | 74 M      | ~1 GB          | ~7x              |
| small   | 244 M     | ~2 GB          | ~4x              |
| medium  | 769 M     | ~5 GB          | ~2x              |
| large   | 1550 M    | ~10 GB         | 1x               |
| turbo   | 809 M     | ~6 GB          | ~8x              |

## 🐛 Solución de Problemas

### Errores Comunes

1. Si obtienes un error de CUDA, verifica que:
   - Tienes los drivers de NVIDIA instalados
   - PyTorch está instalado con soporte CUDA
   - Tu GPU es compatible con CUDA

2. Si ffmpeg no es reconocido:
   - Asegúrate de que está instalado correctamente
   - Verifica que está en el PATH del sistema

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.