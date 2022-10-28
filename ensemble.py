import torch
from PIL import Image
from typing import List
import os
import numpy as np
import os
import json

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble as WLE

from ens_constituents.NVIDIANet import NVIDIANet

CONSTITUENTS_FOLDER = "constituent_models"
ENS_MODELS_FOLDER = "ensemble_models"
CLASS_NAMES = "LOC_synset_mapping.json"

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
    
    
    
    def get_top_predictions(self, image_paths: List[str], combining_method: str="logreg_torch", coupling_method: str="m2", constituents: List[str]=None, n=5):
        constit_outputs, ens_output = self.predict(image_paths=image_paths, combining_method=combining_method, coupling_method=coupling_method, constituents=constituents)
        class_names = np.array(json.load(open(CLASS_NAMES, "r")))
        
        def top_preds(pred: torch.tensor):
            top_preds = torch.topk(pred, k=n, dim=-1)
            sampls = pred.shape[0]
            ret = [["{}: {}".format(class_names[top_preds.indices[sid, i]], top_preds.values[sid, i]) for i in range(n)] for sid in range(sampls)]
            return ret

        constit_top_outputs = {constit: top_preds(pred=constit_outputs[constit]) for constit in constit_outputs}
        ens_top_outputs = top_preds(pred=ens_output)
        
        return constit_top_outputs, ens_top_outputs

        
        