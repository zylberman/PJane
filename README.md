# Análisis de Sentimiento del Mercado con LangGraph y LLMs

Este proyecto utiliza LangGraph y modelos de lenguaje locales para analizar el sentimiento del mercado a partir de noticias obtenidas en la web.

## 🚀 Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/zylberman/PJane
   cd PJane
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate     # En Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno:
   Copia el archivo `.env.example` y renómbralo a `.env`, luego edítalo con tus claves API.
   ```bash
   cp .env.example .env
   ```

## 🛠 Uso

Ejecuta la aplicación:
```bash
python app.py
```

## 📁 Estructura del Proyecto
```
📂 tu-repositorio
├── app.py               # Código principal del flujo de trabajo
├── requirements.txt     # Dependencias del proyecto
├── .env.example         # Variables de entorno de ejem
├── .gitignore           # Archivos y carpetas a ignorar en Git
├── README.md            # Documentación del proyecto
```

## 📌 Notas
- **NO subas el archivo `.env`** con tus claves API.
- **Asegúrate de instalar las dependencias** antes de ejecutar el script.

## 📜 Licencia
Este proyecto está bajo la licencia MIT.

