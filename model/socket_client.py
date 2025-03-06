import socket

# Server setup
host = '127.0.0.1'  # Server's IP address
port = 12346         # The same port as the server

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

# Send messages to the server
while True:
    message = input("Enter message (or 'exit' to quit): ") 
    if message.lower() == 'exit':
        break
    client_socket.send(message.encode())


    message = client_socket.recv(1024)

    print(message.decode())

client_socket.close()
