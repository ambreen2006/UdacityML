import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from PIL import Image

import json

class FlowerClassifier():

    def __init__(self, checkpoint_path, mapping_file_path, use_gpu):

        self.checkpoint_file = checkpoint_path
        self.category_names_file = mapping_file_path
        self.gpu = use_gpu

        self.initialize_transforms()
        self.restore()
        self.load_category_names()

    def initialize_transforms(self):
        m = [0.485, 0.456, 0.406]
        s = [0.229, 0.224, 0.225]
        
        self.data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean = m, 
                                                                        std = s)])

        self.valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = m, 
                                                           std = s)])


    def category_name(self, category_id):
        if category_id in self.cat_to_name:
            return self.cat_to_name[category_id]
        else:
            print("Either category to name json not available to key not in it")
            return ""

    def load_category_names(self):
        self.cat_to_name = []
        if self.category_names_file:
            with open(self.category_names_file, 'r') as f:
                self.cat_to_name = json.load(f)

    def restore(self):
        self.model = torch.load(self.checkpoint_file)
        self.model = self.model.to("cuda" if self.gpu else "cpu")
        self.model.eval()
        self.idx_to_classes = {x: y for y, x in self.model.class_to_idx.items()}

    def process_image(self, image_path):
        image = Image.open(image_path)
        image = self.test_transforms(image).unsqueeze_(0)
        return image.to("cuda" if self.gpu else "cpu")

    def topk(self, image_path, k, use_gpu):
        img = self.process_image(image_path, use_gpu)

        with torch.no_grad():
            output = self.model(img)
            output_ps = torch.exp(output)
            prob, indices = output_ps.topk(k)

        prob = prob.numpy()[0]
        indices = indices.numpy()[0]

        flower_classes = [self.idx_to_classes[x] for x in indices]
        flower_names = [self.cat_to_name[x] for x in flower_classes]
        print(flower_names, prob)

    def predict(self, image_path):
        img = self.process_image(image_path)

        with torch.no_grad():
            output = self.model(img)
            output_ps = torch.nn.functional.softmax(output, dim = 1)
            prob, class_index = torch.max(output_ps, 1)
 
        prob = prob.cpu().numpy().item()
        idx = class_index.cpu().numpy().item()
        flower_class = self.idx_to_classes[idx]

        return (prob, flower_class, self.category_name(flower_class))

class ClassifierInTraining(FlowerClassifier):

    def __init__(self, data_directory, arch, use_gpu):
        self.supported_models = {
            "vgg16" : models.vgg16,
            "vgg11" : models.vgg11,
            "densenet121": models.densenet121
        }

        if not arch in self.supported_models:
            raise Exception('Unsupported arch')

        if not data_directory:
            raise Exception('Data directory for training must be provided')

        self.data_directory = data_directory
        self.arch = arch
        self.gpu = use_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.gpu == True else "cpu")

        self.out_features = 102
        self.learning_rate = 0.0003
        self.epochs = 5
        self.drop_rate = 0.05
        
        self.initialize_transforms()
        self.create_model(self.arch)

    def is_training(self):
        return self.model.training

    def create_model(self, arch):
        self.model = self.supported_models[arch](pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
                                              nn.Linear(in_features, 512),
                                              nn.ReLU(),
                                              nn.Dropout(self.drop_rate),
                                              nn.Linear(512, self.out_features),
                                              nn.LogSoftmax(dim=1)
                                              )
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.model.classifier.parameters()), 
                                    lr = self.learning_rate)
        self.model = self.model.to(self.device)

    def create_data_loaders(self):
        train_datasets = datasets.ImageFolder(self.data_directory  + '/train', transform=self.data_transforms)
        validation_datasets = datasets.ImageFolder(self.data_directory + '/valid', transform=self.valid_transforms)
        self.data_loaders = DataLoader(train_datasets, batch_size=64, shuffle=True)
        self.validation_loaders = DataLoader(validation_datasets, batch_size=32)        

    def train(self):
        self.create_data_loaders()

        running_loss = 0.0
        validation_loss = 0.0

        print("Starting training on {}".format("GPU" if self.gpu else "CPU"))
        
        for e in range(0, self.epochs):
            self.model.train()
            #train
            for inputs, labels in self.data_loaders:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            #validate
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.validation_loaders:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                
                validation_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Size of testloader: {len(testloader)}")
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss:.3f}.. "
                      f"Validation loss: {tvalidation_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0.0
