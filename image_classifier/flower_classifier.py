import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from PIL import Image

import json

class ImageRecognizer():

    def __init__(self, checkpoint_path, mapping_file_path,  use_gpu):
        self.cat_to_name = []
        self.gpu = use_gpu
        self.initialize_transforms()

        if checkpoint_path:
            self.checkpoint_file = checkpoint_path
            self.restore()

        if mapping_file_path:
            self.category_names_file = mapping_file_path
            self.load_category_names()

    def supported_models(self):
        return ({ "vgg16" : models.vgg16, "vgg11" : models.vgg11})        

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
        if self.category_names_file:
            with open(self.category_names_file, 'r') as f:
                self.cat_to_name = json.load(f)

    def restore(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']

        self.model = self.model.to("cuda" if self.gpu else "cpu")
        self.model.eval()
        self.idx_to_classes = {x: y for y, x in self.model.class_to_idx.items()}

    def process_image(self, image_path):
        image = Image.open(image_path)
        image = self.valid_transforms(image).unsqueeze_(0)
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

class ImageClassiferForTraining(ImageRecognizer):

    class Network(nn.Module):
        
        def __init__(self, in_features, hidden_units, out_features, drop_rate):
            super().__init__()

            '''
            # Input to a hidden layer
            self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hidden_units[0])])
        
            # Add a variable number of more hidden layers
            layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
            self.output_layer = nn.Linear(hidden_units[-1], out_features)
            '''
            
            self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hidden_units[0])])
            
            for i in range(0, len(hidden_units)-1):
                self.hidden_layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

            self.output_layer = nn.Linear(hidden_units[-1], out_features)
            self.dropout = nn.Dropout(p = drop_rate)

            print(self.hidden_layers)
            print(self.output_layer)
    

        def forward(self, x):
            for layer in self.hidden_layers:
                x = nn.functional.relu(layer(x))
                x = self.dropout(x)
            x = self.output_layer(x)
            return nn.functional.log_softmax(x, dim = 1)
            
    def __init__(self, data_directory, attributes):
        super(ImageClassiferForTraining, self).__init__(None, None, False)
        if not data_directory:
            raise Exception('Data directory for training must be provided')

        self.data_directory = data_directory
        self.learning_rate = 0.003
        self.hidden_units = [500]
        self.arch = 'vgg16'
        self.gpu = False
        self.save_directory = None

        if 'learning_rate' in attributes:
            self.learning_rate = attributes['learning_rate']
        if 'hidden_layers' in attributes:
            self.hidden_units = attributes['hidden_layers']
        if 'arch' in attributes:
            self.arch = attributes['arch']
            supported_archs = self.supported_models()
            if not self.arch in supported_archs:
                raise Exception('Unsupported arch')
        if 'use_gpu' in attributes:
            self.gpu = attributes['use_gpu']
        if 'save_dir' in attributes:
            self.save_directory = attributes['save_dir']

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.gpu == True else "cpu")

        self.out_features = 102
        self.drop_rate = 0.05
        self.epoch = 0

        self.initialize_transforms()
        self.create_model()

    def is_training(self):
        return self.model.training

    def save(self):        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'model_arch':self.arch,
            'model':self.model
        }, self.save_directory)

           
    def create_model(self):

        base = self.supported_models()[self.arch]
        self.model = base(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[0].in_features
        
        self.model.classifier = self.Network(in_features, self.hidden_units, self.out_features, self.drop_rate)
    
        self.criterion = nn.NLLLoss()

        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.learning_rate)
        
        self.model = self.model.to(self.device)


    def create_data_loaders(self):
        train_datasets = datasets.ImageFolder(self.data_directory  + '/train', transform=self.data_transforms)
        validation_datasets = datasets.ImageFolder(self.data_directory + '/valid', transform=self.valid_transforms)
        
        self.data_loaders = DataLoader(train_datasets, batch_size=64, shuffle=True)
        self.validation_loaders = DataLoader(validation_datasets, batch_size=64)

        self.model.class_to_idx = train_datasets.class_to_idx

    def train(self, epochs):
        self.create_data_loaders()

        running_loss = 0.0

        print("Starting training on {}".format("GPU" if self.gpu else "CPU"))
        print("Epoch from {} to {}".format(self.epoch, epochs))
        
        for e in range(self.epoch, epochs):
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
                validation_loss = 0.0
                accuracy = 0.0

                for inputs, labels in self.validation_loaders:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                    logps = self.model.forward(inputs)
                    loss = self.criterion(logps, labels)
                
                    validation_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e + 1}/{epochs}.. "
                  f"Train loss: {running_loss/len(self.data_loaders):.3f}.. "
                  f"Validation loss: {validation_loss/len(self.validation_loaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(self.validation_loaders)*100:.3f}")
            running_loss = 0.0
        if self.save_directory:
            self.save()

