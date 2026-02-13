# data/base_loader.py

import numpy as np

class BaseLoader:
    """
    Clase base para loaders de datasets.
    Define la interfaz que deben implementar todos los loaders.
    """

    def load(self):
        """
        Debe ser implementado por cada loader específico.
        Devuelve:
            X: np.ndarray o lista de features
            y: np.ndarray o lista de etiquetas
            meta: dict con información adicional (opcional)
        """
        raise NotImplementedError("El método load() debe ser implementado por la subclase")
