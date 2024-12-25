import torch

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