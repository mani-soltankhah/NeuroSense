import torch


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    return model
