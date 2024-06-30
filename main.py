import torchvision
import torch
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass

model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
model.eval()

def load_image(path:str) -> torch.Tensor:
    img = Image.open(path)

    return transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
    ])(img)

image_tensor = load_image("input.jpg")

@dataclass
class Params:
    target: int
    learning_rate: float
    weight_decay: float
    num_iter: int


default_params = Params(0, 1e-2, 100, 200)

def train(img:torch.Tensor, params:Params) -> torch.Tensor:

    noise = torch.zeros_like(img, requires_grad=True)

    opt = torch.optim.SGD([noise], weight_decay=params.weight_decay, lr=params.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        out = model(img.unsqueeze(0))[0]
        loss = loss_fn(out, torch.tensor(params.target))
        print(loss)

    for i in range(params.num_iter):
        perturbed = img + noise
        out = model(perturbed.unsqueeze(0))[0]
        loss = loss_fn(out, torch.tensor(params.target))
        print("iter", i, loss)
        loss.backward()
        opt.step()
    
    return noise
    
def eval(img:torch.Tensor, noise:torch.tensor, params:Params):
    with torch.no_grad():
        out_orig = model(img.unsqueeze(0))[0]
        perturbed = img + noise
        out_perturbed = model(perturbed.unsqueeze(0))[0]
        prob_orig = torch.nn.functional.softmax(out_orig, dim=0)
        prob_perturbed = torch.nn.functional.softmax(out_perturbed, dim=0)
        top_orig = torch.topk(prob_orig, 5)
        top_perturbed = torch.topk(prob_perturbed, 5)
        print(top_orig)
        print(top_perturbed)
        print(prob_perturbed[params.target])
        print(prob_perturbed[top_orig.indices])