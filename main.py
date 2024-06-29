import torchvision
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
model.eval()
img = Image.open("input.jpg")

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
])

image_tensor = preprocess(img)

noise = torch.zeros_like(image_tensor, requires_grad=True)

opt = torch.optim.SGD([noise], weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

target_class = 0
with torch.no_grad():
    out = model(image_tensor.unsqueeze(0))[0]
    loss = loss_fn(out, torch.tensor(target_class))
    print(loss)

n_iter=100
for i in range(n_iter):
    perturbed = image_tensor + noise
    out = model(perturbed.unsqueeze(0))[0]
    loss = loss_fn(out, torch.tensor(target_class))
    print("iter", i, loss)
    loss.backward()
    opt.step()
    
with torch.no_grad():
    perturbed = image_tensor + noise
    out = model(perturbed.unsqueeze(0))[0]
    loss = loss_fn(out, torch.tensor(target_class))
    print(loss)