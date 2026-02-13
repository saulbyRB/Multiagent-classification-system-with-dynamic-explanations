# data/registry.py

from sklearn.model_selection import train_test_split

class DatasetRegistry:
    """
    Registro central de datasets.
    Permite registrar datasets y acceder a ellos para entrenamiento/test.
    """

    def __init__(self):
        self.datasets = {}

    def register(self, dataset_id, loader):
        """
        Registra un dataset.
        loader: función o clase que tenga método load() que devuelva X, y, meta
        """
        self.datasets[dataset_id] = loader

    def list_datasets(self):
        return list(self.datasets.keys())

    def load(self, dataset_id):
        """
        Carga el dataset registrado.
        Devuelve: X, y, meta
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} no registrado")
        loader = self.datasets[dataset_id]
        return loader.load()  # debe devolver X, y, meta

    @staticmethod
    def get_train_test(X, y, test_size=0.2, random_state=42):
        """
        Divide X, y en train/test.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
