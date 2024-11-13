from ollama import Client

# Crear cliente
cliente = Client(host='http://localhost:11434')

# Probar el modelo
respuesta = cliente.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': '¡Hola! ¿Cómo estás?'
    }
])

print(respuesta['message']['content'])