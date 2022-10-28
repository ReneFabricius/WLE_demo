from ens_constituents.Constituent import EnsembleConstituent
import torch
import torchvision.transforms as transforms

class NVIDIANet(EnsembleConstituent):
    def __init__(self, architecture: str, device: torch.device):
        super().__init__(name=architecture, device=device)
        self.arch_ = architecture
        self.model_ = torch.hub.load(
            repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
            model=architecture,
            pretrained=True).eval().to(device)
        self.transform_ = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def predict(self, images: torch.tensor):
        """Performs inference

        Args:
            images (torch.tensor): Batch of preprocessed.
        """
        with torch.no_grad():
            output = self.model_(images)
            
        return output        
    
    def predict_raw(self, images: list):
        """Performs preprocessing and inference

        Args:
            images (list): List of PIL images.
        """
        with torch.no_grad():
            images = torch.cat(self.transform_(images))
            output = self.model_(images)
            
        return output
        