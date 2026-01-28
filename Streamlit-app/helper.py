from PIL import Image
import torch
import torch.nn as nn

from torchvision import models, transforms


model = None
num_classes=6
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage ', 'Rear Crushed ','Rear Normal']

# Load the pre-trained ResNet training_model
class CarResNet(nn.Module):
    def __init__(self,num_classes=num_classes,dropout_rate=0.50):
        super().__init__()

        # Load pre-trained ResNet-50
        self.model = models.resnet50(weights="DEFAULT")

        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4(last convolutional block) and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self,x):
        return self.model(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image_tensor = image.unsqueeze(0)

    global model

    if model is None:
        model = CarResNet(num_classes=num_classes,dropout_rate=0.50)
        model.load_state_dict(torch.load("model\car_damage_resnet50.pth"))
        model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]

