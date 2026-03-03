import torch
import torch.nn as nn

def load_dino_backbone(name="dinov2_vits14_reg"):
    """_summary_

    Args:
        name (str, optional): _description_. Defaults to "dinov2_vits14_reg".
        Other Backbones:
         - No Registers:
            - 'dinov2_vits14'
            - 'dinov2_vitb14'
            - 'dinov2_vitl14'
            - 'dinov2_vitg14'
         - With Registers:
            - 'dinov2_vits14_reg'
            - 'dinov2_vitb14_reg'
            - 'dinov2_vitl14_reg'
            - 'dinov2_vitg14_reg'
            
    """

    return torch.hub.load('facebookresearch/dinov2', name)

class DINOv2_Backbone(nn.Module):
    def __init__(self, backbone="dinov2_vits14_reg"):
        super().__init__()
        self.encoder = load_dino_backbone(backbone)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.encoder(x)
    
class PairwiseDistanceNet(nn.Module):
    def __init__(self, n_classes=5, backbone="dinov2_vits14_reg"):
        super().__init__()
        self.backbone = DINOv2_Backbone(backbone)
        self.n_classes = n_classes
        if (n_classes == None):
            self.x1 = nn.Linear(2*384, 512)
            self.a1 = nn.ReLU()
            self.x2 = nn.Linear(512, 128)
            self.a2 = nn.ReLU()
            self.fc = nn.Linear(128, 1)
        else:
            self.fc = nn.Linear(2 * 384, n_classes)
    
    def forward(self, x1, x2):
        assert self.n_classes is None
        feats = torch.cat([self.backbone(x1), self.backbone(x2)], dim=1)
        feats = self.a1(self.x1(feats))
        feats = self.a2(self.x2(feats))
        return self.fc(feats)

if __name__ == "__main__":
    model = PairwiseDistanceNet(n_classes=None)
    print(model)
