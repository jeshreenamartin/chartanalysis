
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 5)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify_retina(img_path):
    image = Image.open(img_path).convert('RGB')
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_class = np.argmax(probs)
    labels = ["No_DR","Mild","Moderate","Severe","Proliferative_DR"]
    return {"predicted_class": labels[pred_class], "confidence_scores": {labels[i]: float(probs[i]) for i in range(5)}}


print(classify_retina("./datasets/sample_retina.png"))
