import torch


class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, image):
        self.model.eval()
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prediction = torch.sigmoid(output)
            prediction = (prediction > 0.5).float()
        return prediction
