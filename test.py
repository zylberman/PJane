import app

pair = "BTC/USDT"
fecha_str = "2024-12-06"

try:
    response = app.obtener_sentimiento(pair, fecha_str)
    print("Respuesta de app.obtener_sentimiento:", response)
except Exception as e:
    print("Error al probar app.obtener_sentimiento:", e)

