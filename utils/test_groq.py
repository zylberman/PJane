import os
from dotenv import load_dotenv
from groq import Groq

# Cargar variables de entorno
load_dotenv()

# Configurar el cliente de Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Función para enviar una consulta a Groq
def chat_with_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente útil."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",  # Asegúrate de que este modelo esté disponible para ti
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Ejemplo de uso
if __name__ == "__main__":
    user_input = input("Escribe tu mensaje: ")
    response = chat_with_groq(user_input)
    print("Respuesta del modelo:", response)
