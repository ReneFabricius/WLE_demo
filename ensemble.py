import torch
from PIL import Image
from typing import List
import os

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble as WLE

from ens_constituents.NVIDIANet import NVIDIANet

CONSTITUENTS_FOLDER = "constituent_models"
ENS_MODELS_FOLDER = "ensemble_models"

class WLEDemo:
    """
    """
    
    def __init__(self, device: str):
        self.device_ = device
        torch.hub.set_dir(CONSTITUENTS_FOLDER)
        CONSTIT_NAMES = [
            "nvidia_efficientnet_b0",
            "nvidia_efficientnet_widese_b0",
            "nvidia_resneXt",
            "nvidia_resnet50"]
        self.constituents_ = {name: NVIDIANet(architecture=name, device=device) for name in CONSTIT_NAMES}
        
    def predict(self, image_paths: List[str], combining_method: str="logreg_torch", coupling_method: str="m2", constituents: List[str]=None):
        images = [Image.open(fp=path) for path in image_paths]
        if constituents is None:
            constituents = self.constituents_.keys()
        constituents = sorted(constituents)
        const_outputs = []
        const_outputs_dict = {}
        sm = torch.nn.Softmax(dim=-1)
        for constit in constituents:
            output = self.constituents_[constit].predict_raw(images)
            const_outputs.append(output.unsqueeze(0))
            const_outputs_dict[constit] = sm(output).cpu()
        const_outputs = torch.cat(const_outputs, dim=0)
        
        ens_name = "wle_{}_co_{}".format("+".join(constituents), combining_method)
        wle = WLE(device=self.device_)
        wle.load(os.path.join(ENS_MODELS_FOLDER, ens_name))
        wle_outputs, wle_unc = wle.predict_proba(const_outputs, coupling_method=coupling_method)
        
        return const_outputs_dict, wle_outputs
        
        