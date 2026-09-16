import torch


class Predictor:
    def __init__(self, model, device, threshold):
        self.model = model
        self.device = device
        self.threshold = threshold

    def predict(self, image):
        self.model.eval()
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probability = torch.sigmoid(output)
            prediction = (probability > self.threshold).float()
        return probability, prediction
