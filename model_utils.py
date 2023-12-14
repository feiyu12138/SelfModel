
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def sin_activation(x):
    return torch.sin(30*x)

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
        
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
        # return x if self.is_last else torch.relu(x)

class LinearReluLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.relu(x)


class StateConditionModulatedModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_features=256):
        super(StateConditionModulatedModel, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layerq2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers1 = SirenLayer(in_channels-3, half_hidden_features, is_first=True)
        self.layers2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers3 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers4 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers5 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layerm1 = nn.Linear(half_hidden_features+in_channels-3, hidden_features)
        self.layerm2 = nn.Linear(hidden_features+in_channels-3, hidden_features)
        self.layerm3 = nn.Linear(hidden_features+in_channels-3, hidden_features)
        self.layerm4 = nn.Linear(hidden_features+in_channels-3, hidden_features)
        self.layerm5 = nn.Linear(hidden_features+in_channels-3, hidden_features)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, hidden_features)
        self.layer6 = SirenLayer(hidden_features, hidden_features)
        self.layer7 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        x = self.layerq2(x)
        return x

    def state_encoder(self, x):
        x = self.layers1(x)
        x1 = self.layers2(x)
        x2 = self.layers3(x1) + x
        x3 = self.layers4(x2) + x1
        x4 = self.layers5(x3) + x2
        return x4

    def forward(self, x):
        query_feat = self.query_encoder(x[:, :3])
        state_feat = self.state_encoder(x[:, 3:])
        m_x1 = self.layerm1(torch.cat((state_feat, x[:, 3:]), dim=1))
        m_x2 = self.layerm2(torch.cat((F.relu(m_x1), x[:, 3:]), dim=1)) 
        m_x3 = self.layerm3(torch.cat((F.relu(m_x2),x[:, 3:]),dim=1)) + m_x1
        m_x4 = self.layerm4(torch.cat((F.relu(m_x3),x[:, 3:]),dim=1)) + m_x2
        m_x5 = self.layerm5(torch.cat((F.relu(m_x4), x[:, 3:]), dim=1)) + m_x3
        x1 = torch.cat((query_feat, query_feat), dim=1) 
        x2 = self.layer2(x1 * m_x1)
        x3 = self.layer3(x2 * m_x2) + x1
        x4 = self.layer4(x3 * m_x3) + x2
        x5 = self.layer5(x4 * m_x4) + x3
        x6 = self.layer6(x5 * m_x5) + x4
        x7 = self.layer7(x6)
        return x7
        

class StateConditionMLPQueryModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_features=256):
        super(StateConditionMLPQueryModel, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layers1 = SirenLayer(in_channels-3, half_hidden_features, is_first=True)
        self.layers2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers3 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers4 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x

    def state_encoder(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        return x

    def forward(self, x):
        query_feat = self.query_encoder(x[:, :3])
        state_feat = self.state_encoder(x[:, 3:])
        x = torch.cat((query_feat, state_feat), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class KinematicFeatToLinkModel(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=3, hidden_features=64):
        super(KinematicFeatToLinkModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, hidden_features)
        self.layer2 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class KinematicScratchModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_features=128, hidden_hidden_features=64):
        super(KinematicScratchModel, self).__init__()

        # original self-model's kinematic branch
        self.layer1 = SirenLayer(in_channels, hidden_features, is_first=True)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        # newly added branches for X_link tasks
        self.layer5 = SirenLayer(hidden_features, hidden_hidden_features)
        self.layer6 = SirenLayer(hidden_hidden_features, out_channels, is_last=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x