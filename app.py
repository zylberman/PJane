import os
import re
import json
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from dataclasses import dataclass
from langgraph.graph import END, StateGraph, START  # Importar definiciones del grafo

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Establecer la variable de entorno para la clave de la API de Serper
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

# Configuración de dos modelos de lenguaje que se ejecutan en local a través de LM Studio.
# openai_api_base indica la dirección base del servidor donde corre el modelo
# openai_api_key se usa para autenticar (aquí es un valor ficticio porque se está usando LM Studio local)
llm_qwen = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  
    openai_api_key="lm-studio",  
    model_name="qwen2.5-7b-instruct-1m"
)

llm_deep_seek = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  
    openai_api_key="lm-studio",  
    model_name="deepseek-r1-distill-llama-8b"
)

# Inicializamos una instancia de la herramienta de búsqueda con Google Serper
search = GoogleSerperAPIWrapper()

# Lista de herramientas disponibles para el flujo
tools = [
    Tool(
        name="search_web",
        func=search.run,
        description="Use this tool to search for information on the web."
    )
]

# Definimos la estructura (estado) que queremos manejar con dataclasses,
# en este caso tiene dos campos: 'query' (la consulta a realizar) y 'response' (la respuesta obtenida).
@dataclass
class AgentState:
    query: str
    date: str 
    response: str

# Creamos un grafo de estados usando LangGraph y le indicamos el tipo de estado que maneja.
workflow = StateGraph(AgentState)

# ----------
# Nodo 1: Valida la entrada de información, que sea un activo, divisa, pais o par de divisas valido
# ----------
def validar_activo(state: AgentState) -> AgentState:
    patron_activo = r"^[A-Z]{6,7}$|^(Bitcoin|Ethereum|Oro|Petróleo|España|EE.UU|China)$"

    while True:  # Bucle que repite la solicitud hasta que los datos sean correctos
        query = state.query
        date = state.date

        # Validar activo
        if not re.match(patron_activo, query, re.IGNORECASE):
            print(f"❌ Activo inválido: {query}")
            query = input("Ingrese un activo válido (Ejemplo: BTCUSDT, Ethereum, Oro): ").strip()

        # Validar fecha
        try:
            datetime.strptime(date, "%Y-%m-%d")  # Verifica que la fecha tenga el formato correcto
        except ValueError:
            print(f"❌ Fecha inválida: {date}")
            date = input("Ingrese una fecha válida en formato YYYY-MM-DD: ").strip()
            continue  # Si la fecha es inválida, vuelve a solicitarla

        # Si ambos valores son correctos, salimos del bucle
        if re.match(patron_activo, query, re.IGNORECASE):
            break

    print(f"✅ Activo válido detectado: {query}")
    print(f"✅ Fecha válida detectada: {date}")

    return AgentState(query=query, date=date, response="valid")
 

# ----------
# Nodo 2: Ejecutar búsqueda de noticias en la web
# ----------
from datetime import datetime

def buscar_noticias(state: AgentState) -> AgentState:
    # Obtener la fecha actual en formato YYYY-MM-DD
    fecha_actual = state.date

    # Obtener una fecha específica (por ejemplo, 1 de enero de 2025)
    # fecha_actual = datetime(2025, 1, 1).strftime('%Y-%m-%d')

    # Incluir la fecha en la búsqueda para obtener noticias recientes
    query = f"Noticias recientes sobre {state.query} {fecha_actual}"
    
    print(f"🔍 Buscando en la web con SERPER: {query}")

    # Llamar a SERPER con la consulta que incluye la fecha
    search_results = search.run(query)

    # Verificar si SERPER devuelve resultados
    if not search_results:
        print("⚠ SERPER no devolvió ningún resultado.")
    else:
        print("📌 Resultados obtenidos de SERPER:\n", search_results)

    return AgentState(query=state.query, date=state.date, response=search_results)


# ----------
# Nodo 3: Evalua la calidad de las noticias
# ----------
def analizar_noticias(state: AgentState) -> AgentState:
    if not state.response or state.response == "invalid":
        return AgentState(query=state.query, date=state.date, response="No se encontraron noticias relevantes.")
    
    print("📊 Analizando noticias con LLM...")

    # 1. Unir las noticias en un solo texto, si es una lista
    noticias_texto = "\n".join(state.response) if isinstance(state.response, list) else state.response

    # 2. Construir el prompt
    prompt = f"""Analiza el sentimiento de las siguientes noticias y clasifícalo como positivo, negativo o neutral.
    Devuelve la respuesta en formato JSON con estas claves:
    {{
      "resumen": "Descripción general...",
      "sentimientos": [
        {{
          "noticia": "...",
          "sentimiento": "positivo|negativo|neutral"
        }},
        ...
      ]
    }}
    {noticias_texto}
    """

    # 3. Invocar al LLM
    respuesta_modelo = llm_qwen.invoke([HumanMessage(content=prompt)])
    print("Respuesta de LLM (raw):", respuesta_modelo.content)

    # 4. Limpiar la respuesta del modelo para eliminar el bloque de código Markdown
    respuesta_limpia = respuesta_modelo.content.strip("```json\n").strip("\n```")

    # 5. Parsear la respuesta JSON
    try:
        resultado_json = json.loads(respuesta_limpia)
    except json.JSONDecodeError:
        print("⚠ Error parseando el JSON devuelto por LLM.")
        resultado_json = {
            "resumen": "No se pudo parsear correctamente.",
            "sentimientos": []
        }
    
    # 6. Generar un texto con cada noticia y su sentimiento
    noticias_presentadas = []
    for item in resultado_json.get("sentimientos", []):
        noticia = item.get("noticia", "")
        sentimiento = item.get("sentimiento", "")
        noticias_presentadas.append(f"• Noticia: {noticia}\n  Sentimiento: {sentimiento}")

    # Unir en un solo bloque de texto
    noticias_format = "\n".join(noticias_presentadas)

    # 7. Este nodo retorna un estado donde .response = noticias mostradas
    #    Pero conservamos la data estructurada en, por ejemplo, un nuevo campo
    data_para_el_nodo4 = {
        "resumen_llm": resultado_json.get("resumen", ""),
        "sentimientos_list": resultado_json.get("sentimientos", [])
    }

    # 8. Crear un texto final para "response" que muestre las noticias
    texto_final = (
        "📄 Análisis de Noticias:\n"
        f"{noticias_format}\n"
        f"\nResumen LLM: {resultado_json.get('resumen','No hay resumen.')}"
    )

    # 9. Devolver un AgentState
    return AgentState(
        query=state.query,
        date=state.date,
        response={
            "presentation": texto_final,
            "structured_data": data_para_el_nodo4
        }
    )

# ----------
# Nodo 4: Resumir sentimiento del mercado
# ----------
def resumir_sentimiento(state: AgentState) -> AgentState:
    # 1. Recibimos un diccionario en state.response
    data = state.response
    if not isinstance(data, dict):
        return AgentState(query=state.query, response="No se pudo resumir el sentimiento.")

    structured_data = data.get("structured_data", {})
    sentimientos_list = structured_data.get("sentimientos_list", [])

    # 2. Contar positivos, negativos, neutrales
    positivos = sum(1 for s in sentimientos_list if "positivo" in s.get("sentimiento","").lower())
    negativos = sum(1 for s in sentimientos_list if "negativo" in s.get("sentimiento","").lower())
    neutrales = sum(1 for s in sentimientos_list if "neutral"  in s.get("sentimiento","").lower())

    # 3. Determinar la tendencia global
    if positivos > negativos:
        tendencia = "📈 Positiva"
    elif negativos > positivos:
        tendencia = "📉 Negativa"
    else:
        tendencia = "⚖ Neutral"

    # 4. Construir el texto final
    resumen_global = {
        "query": state.query,
        "date": state.date,
        "tendencia": tendencia,
        "positivos": positivos,
        "negativos": negativos,
        "neutrales": neutrales,
        "resumen_llm": structured_data.get("resumen_llm", "No hay resumen"),
        "detalles": data.get("presentation", "No hay detalle de noticias"),
    }

    # 5. Devolver un nuevo AgentState
    return AgentState(query=state.query, date=state.date, response=resumen_global)

# Definir edges y flujo de control
workflow.add_edge(START, "validate")
workflow.add_edge("summary", END)

# Añadimos los nodos definidos al grafo de estados
workflow.add_node("validate", validar_activo)
workflow.add_node("search", buscar_noticias)
workflow.add_node("analyze", analizar_noticias)
workflow.add_node("summary", resumir_sentimiento)

workflow.add_edge("validate", "search")
# Reemplazar la arista que iba de "search" -> "evaluate" a "search" -> "analyze"
workflow.add_edge("search", "analyze")

# Conectar "analyze" -> "summary"
workflow.add_edge("analyze", "summary")


# Condición: si la validación es exitosa, pasa a "search", de lo contrario, se queda en "validate"
workflow.add_conditional_edges("validate", lambda state: "search" if state.response == "valid" else "END")

# Compilar el grafo
graph = workflow.compile()

# ----------
# Ejecución del flujo
# ----------
def obtener_sentimiento(query: str, date: str) -> dict:
    # Inicializar el estado del agente
    initial_state = AgentState(query=query, date=date, response="")

    # Ejecutar el flujo
    result = graph.invoke(initial_state)

    print("\n🔍 Resumen del Sentimiento del Mercado:",result)
    # Retornar la respuesta final
    return result["response"]

# query = "BTCUSDT"
# result = graph.invoke(AgentState(query=query, date=datetime.now(), response=""))
# print("\n✅ Respuesta final:", result["response"])
# Permitir ejecución desde otro script
if __name__ == "__main__":
    # Solicitar activo con un valor por defecto
    query = input("Ingrese el activo a consultar (por defecto BTCUSDT): ").strip()
    if not query:
        query = "BTCUSDT"  # Valor por defecto si no se ingresa nada

    # Solicitar fecha con un valor por defecto
    date = input("Ingrese la fecha en formato YYYY-MM-DD (por defecto hoy): ").strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")  # Fecha actual por defecto

    # Validar que la fecha ingresada tenga el formato correcto
    try:
        datetime.strptime(date, "%Y-%m-%d")  # Verifica que la fecha sea válida
    except ValueError:
        print("❌ Error: La fecha ingresada no es válida. Use el formato YYYY-MM-DD.")
        exit(1)  # Sale del programa si la fecha es inválida
    
    # Ejecutar el flujo con los valores validados
    resultado = obtener_sentimiento(query, date)

    # Mostrar solo el análisis final en texto
    print("\n✅ Respuesta final:\n")
    print(resultado["detalles"])

    # Mostrar la tendencia correctamente
    print("\n✅ Tendencia:\n")
    print(resultado["tendencia"])

# Permitir ejecución desde otro script
"""
import app

query = "BTCUSDT"
date = "2025-02-24"  # Puedes cambiarlo a la fecha actual

resultado = app.obtener_sentimiento(query, date)

print("\n🔍 Resumen del Sentimiento del Mercado:")
print(resultado)
"""
