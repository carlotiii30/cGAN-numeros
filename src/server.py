import socket
import threading
from handler import Handler


class Server:
    """Class that represents a server that listens on a port and handles
    client connections.

    This class is responsible for starting a server that listens on a specific
    port and handles incoming client connections.

    Attributes:
        host (str): IP address or host name where the server will listen.
        port (int): Port where the server will listen.
    """

    def __init__(self, host, port):
        """Class constructor.

        Args:
            host (str): IP address or host name where the server will listen.
            port (int): Port where the server will listen.
        """
        self.host = host
        self.port = port

    def start(self):
        """Method that starts the server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Server listening on port {self.port}")
            while True:
                client_socket, _ = server_socket.accept()
                threading.Thread(
                    target=self.client_handler, args=(client_socket,)
                ).start()

    def client_handler(self, client_socket):
        """Method that handles a connection with a client.

        Args:
            client_socket (socket): Client socket.
        """
        handler = Handler(client_socket)
        handler.handle()
