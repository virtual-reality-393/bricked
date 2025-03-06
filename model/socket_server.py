import socket
from assistant import VoiceAssistant
# Server setup
host = '127.0.0.1'  # Localhost
port = 12346         # Port to listen on

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

assistant = VoiceAssistant()

print(f"Server listening on {host}:{port}...")

# Accept client connection
client_socket, client_address = server_socket.accept()
print(f"Connection established with {client_address}")

# Receive and print messages from client
while True:
    data = client_socket.recv(1024)
    
    if not data:
        print("No data was sent, shutting down")
        break  # If no data is received, break the loop

    message = data.decode()
    print("Received:",message)

    response = assistant.generate_response(message)
    print("Response:",response)
    client_socket.send(response.encode())

client_socket.close()
server_socket.close()
