import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(parameters, name, optimizer_args):
        
    if name == "adam":
        optimizer = torch.optim.Adam(parameters, **optimizer_args)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    elif name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **optimizer_args)
    elif name == "radam":
        optimizer = torch.optim.RAdam(parameters, **optimizer_args)
    else:
        return NotImplementedError
    
    return optimizer

class LinkPredictor(nn.Module):
    r"""LinkPredictor for graph link prediction task.

    Parameters
    ----------
    n_hid: int
        Input size.
    n_out: int
        Output size.

    """
    def __init__(self, n_hid, n_out):
        super(LinkPredictor, self).__init__()

        self.fc1 = nn.Linear(n_hid * 2, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)

    def forward(self, src, dst):
        x = torch.cat([src, dst], 1)
        y = self.fc2(F.relu(self.fc1(x)))
        return y