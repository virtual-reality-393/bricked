import socket

# Server setup
host = '192.168.1.103'  # Localhost
port = 12346         # Port to listen on

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Server listening on {host}:{port}...")

# Accept client connection
client_socket, client_address = server_socket.accept()
print(f"Connection established with {client_address}")

# Receive and print messages from client
while True:
    data = client_socket.recv(1024)
    if not data:
        break  # If no data is received, break the loop
    print(f"Received message: {data.decode()}")

client_socket.close()
server_socket.close()
