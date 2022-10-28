from abc import ABC, abstractclassmethod, abstractmethod
import torch

class EnsembleConstituent(ABC):
    @abstractmethod
    def __init__(self, name: str, device: torch.device):
        self.name_ = name
        self.device_ = device
    
    @abstractmethod
    def predict(self, images: list):
        pass

