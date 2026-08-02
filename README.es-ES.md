

<p align="center">
  <img src="/static/qwentts.webp" alt="Qwen TTS WebUI" width="300">
</p>

# Qwen TTS WebUI ![icon](/static/qwentts.ico)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![uv](https://img.shields.io/badge/uv-enabled-green.svg)

Genera voz con IA usando Qwen TTS: sube un audio de referencia y crea nuevas síntesis de voz en una interfaz web limpia y moderna.

## Características

- **Clonación de voz** - Sube un audio de referencia o graba directamente en el navegador
- **Soporte multilingüe** - Genera voz en múltiples idiomas
- **Transcripción automática** - Transcripción automática usando Whisper
- **Gestión de audio** - Organiza y descarga los archivos de audio generados
- **YT-DLP** - Descarga audios de referencia directamente desde un enlace de YouTube
- **Nombres personalizados** - Renombra los audios de referencia para una mejor organización

## Inicio Rápido

### Requisitos previos

- **GPU compatible con CUDA** con ~6 GB de VRAM (recomendado)
- **Nix o Python +3.10**
- **uv** (administrador de paquetes, recomendado para entornos reproducibles)

### Instalación y Configuración

<details>
<summary>Recomendado: Usando Nix (Reproducible)</summary>

```bash
# Clonar el repositorio
git clone https://github.com/AfkaraLP/qwen3-tts-webui.git
cd qwen3-tts-webui

# Entrar al entorno de desarrollo
nix develop
uv sync

# Iniciar el servidor
uv run start_server.py
# Consulta localhost:8000/docs para la documentación de la API
```

</details>

<details>
<summary>📦 Alternativa: Usando solo uv</summary>

```bash
# Clonar el repositorio
git clone https://github.com/AfkaraLP/qwen3-tts-webui.git
cd qwen3-tts-webui

# Instalar dependencias
uv sync

# Iniciar el servidor
uv run start_server.py
```

</details>

### Acceder a la Interfaz Web

Abre **http://localhost:8000** en tu navegador para acceder a la interfaz web.

## 🎯 Guía de Uso

### 1. Usar una Referencia Existente
1. Selecciona un audio de referencia previamente subido desde el menú desplegable
2. Opcionalmente, renómbralo usando el campo de renombrado
3. Ingresa el texto que deseas generar
4. Selecciona el idioma y haz clic en "Clonar Voz"

### 2. Subir Nueva Referencia
1. Elige tu fuente de audio:
   - **Subir Archivo**: Selecciona un archivo de audio desde tu dispositivo
   - **Grabar Audio**: Graba directamente usando tu micrófono
   - **YouTube**: Ingresa una URL de YouTube para extraer el audio
2. Agrega un nombre personalizado para una mejor organización
3. Ingresa el texto y selecciona el idioma
4. Haz clic en "Subir y Clonar"

### 3. Gestionar Audio Generado
- Visualiza todos tus audios generados en la sección "Audios Generados"
- Reproduce el audio directamente en el navegador
- Descarga los archivos para uso sin conexión

## 🛠️ Configuración

### Variables de Entorno

| Variable | Predeterminado | Descripción |
|----------|----------------|-------------|
| `VOICE_CLONER_PORT` | `8000` | Puerto para el servidor web |
| `CUDA_VISIBLE_DEVICES` | `0` | Dispositivo GPU a utilizar |

```bash
# Ejemplo de puerto personalizado
VOICE_CLONER_PORT=3000 python start_server.py

# Usar GPU específica
CUDA_VISIBLE_DEVICES=1 python start_server.py
```

## 🤝 Contribuir

¡Agradecemos las contribuciones! Por favor, asegura la reproducibilidad siguiendo estos pasos:

### Entorno de Desarrollo

Usa **Nix** y **uv** para un entorno de desarrollo consistente:

```bash
# Configurar entorno de desarrollo
nix develop
```

### Pautas de Contribución

1. **Haz un Fork** del repositorio
2. **Crea una rama de características**: `git checkout -b feature/amazing-feature`
3. **Realiza cambios** con mensajes de commit claros
4. **Prueba exhaustivamente**, incluidos los casos extremos
5. **Envía un pull request** con una descripción detallada

## 🙏️ Apoya el Desarrollo

Si encuentras útil este proyecto, considera apoyarme:

<div align="center">
  <a href="https://ko-fi.com/afkaralp" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_stroke.png" alt="Support me on Ko-fi" height="36"/>
  </a>
</div>

¡Tu ayuda contribuye a mantener y mejorar el proyecto! 🚀

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - consulta el archivo [LICENSE](LICENSE.txt) para más detalles.

## 🙋 Preguntas Frecuentes

<details>
<summary>❓ Preguntas Frecuentes</summary>

**P: ¿Qué formatos de audio son compatibles?**
R: Se admiten la mayoría de los formatos de audio comunes (MP3, WAV, M4A, etc.) para la carga.

**P: ¿Qué duración debe tener el audio de referencia?**
R: 10-30 segundos es lo ideal. El sistema recorta automáticamente a un máximo de 60 segundos.

**P: ¿Puedo usar esto con fines comerciales?**
R: Por favor, verifica la licencia y los términos de uso del modelo Qwen TTS para aplicaciones comerciales.

**P: ¿Por qué usar Nix y uv?**
R: Proporcionan **entornos reproducibles**: cualquier persona puede recrear exactamente la misma configuración de desarrollo, asegurando un comportamiento consistente en diferentes máquinas.

</details>
