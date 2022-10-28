
from ens_constituents.NVIDIANet import NVIDIANet

class WLEDemo:
    """
    """
    
    def __init__(self, device):
        self.device_ = device
        CONSTIT_NAMES = [
            "nvidia_efficientnet_b0",
            "nvidia_efficientnet_widese_b0",
            "nvidia_resneXt",
            "nvidia_resnet50"]
        self.constituents_ = {name: NVIDIANet(architecture=name, device=device) for name in CONSTIT_NAMES}
        