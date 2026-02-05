# data/registry.py

from sklearn.model_selection import train_test_split


class DatasetRegistry:
    """
    Registro central de datasets.
    """

    def __init__(self):
        self.datasets = {}

    def register(self, dataset_id, loader):
        self.datasets[dataset_id] = loader
 
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} no registrado")
        return self.datasets[dataset_id].load()

    def list_datasets(self):
        return list(self.datasets.keys())


    def get_train_test(X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
