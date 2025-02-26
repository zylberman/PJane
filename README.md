# 📈 Análisis de Sentimiento del Mercado con LangGraph y LLMs - Implementación con Freqtrade

Este proyecto utiliza LangGraph y modelos de lenguaje locales para analizar el sentimiento del mercado a partir de noticias obtenidas en la web.

Este directorio contiene estrategias personalizadas para el bot de trading **Freqtrade**.

---

## 🚀 Instalación y Configuración

### 1️⃣ Instalar Docker (Si no lo tienes)
Freqtrade se ejecuta dentro de un contenedor **Docker**. Si aún no tienes **Docker**, instálalo:

- **Windows/Mac:** [Descargar Docker Desktop](https://www.docker.com/get-started)
- **Linux (Debian/Ubuntu):**
  
  ```bash
  sudo apt update
  sudo apt install -y docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

Para verificar la instalación de Docker:

```bash
docker --version
```

---

## 🛠 Configuración y Uso

### 2️⃣ Descargar y ejecutar el contenedor de Freqtrade
Para obtener la última versión estable de **Freqtrade**, ejecuta:

```bash
docker pull freqtrade/freqtrade:stable
```

### 3️⃣ Iniciar el contenedor y montar las estrategias
Ejecuta el contenedor y monta el directorio `strategies/` en Freqtrade:

```bash
docker run -it --rm --name freqtrade \
    -v $(pwd):/freqtrade/user_data/strategies \
    freqtrade/freqtrade:stable bash
```

Esto hará que las estrategias dentro de este repositorio se sincronicen con la carpeta `strategies` dentro del contenedor.

### 4️⃣ Activar Freqtrade dentro del contenedor
Una vez dentro del contenedor de Docker, puedes verificar la instalación ejecutando:

```bash
freqtrade --help
```

Para inicializar una configuración básica:

```bash
freqtrade new-config --config user_data/config.json
```

Si deseas ejecutar una simulación en **modo Dry-Run** (sin operar realmente):

```bash
freqtrade trade --dry-run
```

---
## 🚀 Descarga el repositorio en la carpeta strategies de Freqtrade

### 5️⃣ Clonar este repositorio
   ```bash
   git clone https://github.com/zylberman/PJane
   cd PJane
   ```

### 6️⃣ Crear y activar un entorno virtual
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate     # En Windows
   ```

### 7️⃣ Instalar las dependencias
   ```bash
   pip install -r requirements.txt
   ```

### 8️⃣ Configurar las variables de entorno
   Copia el archivo `.env.example` y renómbralo a `.env`, luego edítalo con tus claves API.
   ```bash
   cp .env.example .env
   ```

## 📌 Archivos en este directorio

- `app.py` → Código principal del bot.
- `pj.py` → Script con funciones auxiliares para estrategias personalizadas.
- `requirements.txt` → Dependencias necesarias para las estrategias.
- `sample_strategy.py` → Plantilla de estrategia básica en Freqtrade.
- `instrucciones.txt` → Notas sobre la implementación de estrategias.

---

## 🚀 Instala LMStudio

Para utilizar modelos de lenguaje con Freqtrade, sigue estos pasos:

- Descarga e instala **LMStudio** desde su página oficial.
- Elige un modelo compatible con LMStudio. Se recomienda usar:
  - `qwen2.5-7b-instruct-1m`
  - `deepseek-r1-distill-llama-8b`
- Inicia el servidor de LMStudio y configura `app.py` para usar el modelo descargado.

---

## 🎯 Cómo Crear y Probar Estrategias

### 9️⃣ Probar la estrategia con Backtesting
Para ejecutar pruebas con datos históricos:

```bash
freqtrade backtest --config user_data/config.json --strategy MiEstrategia
```

---

## 🎯 Uso de la estrategia

La estrategia realiza análisis de sentimiento del mercado basado en noticias de la web. Su funcionamiento se divide en los siguientes pasos:

- **Entrada de datos:** El usuario ingresa la divisa que desea evaluar.
- **Recopilación de información:** El script busca noticias y datos relevantes sobre la divisa ingresada.
- **Análisis de sentimiento:** Utiliza modelos de lenguaje para determinar el sentimiento general del mercado.
- **Estrategia de trading:** Basado en el análisis de sentimiento y en indicadores técnicos, la estrategia decide si:
  - **Compra** la divisa.
  - **Vende** la divisa.
  - **Mantiene la posición** si las condiciones no son favorables.

---

## 📜 Licencia
Este proyecto está bajo la licencia MIT.

🚀 ¡Feliz Trading! 🤑

