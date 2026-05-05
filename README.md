# 🎙️ TranscribeAI – Clean Verbatim

TranscribeAI es una aplicación web ligera y 100% local para transcribir archivos de audio y vídeo utilizando **Whisper (faster‑whisper)** con post‑procesado **Clean Verbatim**. Elimina automáticamente muletillas, repeticiones, tartamudeos, sonidos no verbales y ruidos del lenguaje para generar una transcripción fluida, profesional y con marcas de tiempo.

[![Licencia MIT](https://img.shields.io/badge/licencia-MIT-green)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

## ✨ Características

- 🎥 **Soporta audio y vídeo**: MP3, WAV, FLAC, MP4, MKV, MOV, AVI y muchos más (vía FFmpeg).
- 🧹 **Clean Verbatim integrado**: Elimina *ehs*, *o sea*, *pues*, repeticiones, falsos comienzos, toses, risas y otras interjecciones.
- ⏱️ **Marcas de tiempo**: Cada segmento muestra su posición en el audio.
- 📄 **Salida en TXT**: Transcripción lista para usar, copiar o descargar.
- 🌍 **Detección de idioma** o selección manual (12+ idiomas).
- ⚡ **Modelos ajustables**: desde Tiny (ultrarrápido) hasta Large v3 (máxima precisión).
- 🖥️ **Interfaz web moderna** (diseño responsivo, modo claro/oscuro, arrastrar y soltar).
- 🔒 **Privado y sin límites**: Todo el procesado ocurre en tu máquina, no se envían datos a la nube.
- 🐳 **Fácil de desplegar** con Docker o directamente en cualquier servidor Linux/Windows.

## 📋 Requisitos previos

- Python 3.8 o superior
- [FFmpeg](https://ffmpeg.org/) (necesario para procesar vídeos y algunos formatos de audio)
- 4 GB de RAM recomendados (para el modelo `small`)

### Instalación de FFmpeg

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install ffmpeg
