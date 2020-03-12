import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from PIL import Image

import json

class FlowerClassifier():

    def __init__(self):

        m = [0.485, 0.456, 0.406]
        s = [0.229, 0.224, 0.225]
        
        self.data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean = m, 
                                                                        std = s)])

        self.test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = m, 
                                                           std = s)])
        self.load_category_names()

    def load_category_names(self):
        with open('cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)
    
    def restore(self, checkpoint):
        self.model = torch.load(checkpoint)
        self.idx_to_classes = {x: y for y, x in self.model.class_to_idx.items()}

    def process_image(self, image_path, use_gpu):
        image = Image.open(image_path)
        image = self.test_transforms(image).unsqueeze_(0)
        return image.to("cuda" if use_gpu else "cpu")

    def topk(self, image_path, k, use_gpu):

        img = self.process_image(image_path, use_gpu)
        self.model = self.model.to("cuda" if use_gpu else "cpu")

        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
            output_ps = torch.exp(output)
            prob, indices = output_ps.topk(k)

        prob = prob.numpy()[0]
        indices = indices.numpy()[0]
        #print(prob, indices)
        flower_classes = [self.idx_to_classes[x] for x in indices]
        flower_names = [self.cat_to_name[x] for x in flower_classes]
        print(flower_names, prob)
        
    def predict(self, image_path, use_gpu = False):
        img = self.process_image(image_path, use_gpu)

        self.model = self.model.to("cuda" if use_gpu else "cpu")

        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
            output_ps = torch.nn.functional.softmax(output, dim = 1)
            prob, predicted = torch.max(output_ps, 1)

        idx = predicted.numpy().item()
        flower_class = self.idx_to_classes[idx]
        return (prob.numpy(), flower_class, self.cat_to_name[flower_class])
