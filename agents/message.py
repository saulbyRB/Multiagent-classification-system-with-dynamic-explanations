# message.py
class Message:
    """
    Mensaje simple entre agentes simulando SPADE.
    """
    def __init__(self, sender: str, body: dict):
        self.sender = sender
        self.body = body
