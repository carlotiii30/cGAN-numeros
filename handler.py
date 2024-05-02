import json
import logging
import drawing
import utils


class Handler:
    """Class that handles client requests.

    This class handles the requests sent by the client and executes the
    corresponding actions.

    Attributes:
        socket (socket): Client socket.
    """

    def __init__(self, socket):
        """Class constructor.

        Args:
            socket: Client socket.
        """
        self.socket = socket

    def handle(self):
        """Method that handles the client's request."""
        with self.socket:
            data = self.socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                text = request.get("text", "")

                if command == "generate":
                    try:
                        cond_gan = utils.load_model_with_weights("cond_gan_weights.weights.h5")
                        img = drawing.draw_number(text, cond_gan)
                        img = img.tolist()

                        response = {
                            "status": "success",
                            "message": "Image generated successfully",
                            "image": img,
                        }

                        logging.info("Image generated successfully")

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error generating the image: {str(e)}",
                        }

                        logging.error(f"Error generating the image: {str(e)}")
                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown command: {command}",
                    }

                    logging.error(f"Unknown command: {command}")

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "message": f"Error decoding JSON: {str(e)}",
                }

                logging.error(f"Error decoding JSON: {str(e)}")

            except Exception as e:
                response = {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }

                logging.error(f"Error: {str(e)}")

            try:
                self.socket.sendall(json.dumps(response).encode())

                logging.info("Response sent to the client")

            except Exception as e:
                print(f"Error sending the response to the client: {str(e)}")

                logging.error(f"Error sending the response to the client: {str(e)}")
