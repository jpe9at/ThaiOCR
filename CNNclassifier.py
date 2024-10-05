import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class CNNThai(nn.Module):
    def __init__(self, hidden_size=64, output_size=3, optimizer='Adam', learning_rate=0.0001, loss_function='CEL', l1=0.0, l2=0.0, scheduler=None, param_initialisation=None):
        super(CNNThai, self).__init__()
        
        
        # Define convolutional layers
        # One imput channel, because we are dealing with graysacle images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        # intial (40,40) has been halved twice by two pooling layers
        self.fc1 = nn.Linear(64 * 5 * 5, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self._init_weights = 'Default'

        # Initialize optimizer, loss function, and other parameters
        self.optimizer = self.get_optimizer(optimizer, learning_rate, l2)
        self.loss = self.get_loss(loss_function)
        self.learning_rate = learning_rate
        self.l1_rate = l1
        self.l2_rate = l2

        self.scheduler = self.get_scheduler(scheduler, self.optimizer)
        
        if param_initialisation is not None: 
            layer, init_method = param_initialisation 
            self.initialize_weights(layer, init_method)
        self.device = next(self.parameters()).device
        print("Current Device: " + str(self.device))
    
    def forward(self, x):
        # Forward pass through the convolutional layers
        x = x.to(next(self.parameters()).device)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten the tensor
        x = x.view(-1, 64 * 5 * 5)

        # Forward pass through the fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        #Since CEL expects logits we won't apply the softmax here.
        #x = self.log_softmax(x)

        return x

    def get_optimizer(self, optimizer, learning_rate, l2):
        Optimizers = {
            'Adam': optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2), 
            'SGD': optim.SGD(self.parameters(), lr=learning_rate, momentum=0.09, weight_decay=l2)
        }
        return Optimizers[optimizer]

    def l1_regularization(self, loss):
        l1_reg = sum(p.abs().sum() * self.l1_rate for p in self.parameters())
        loss += l1_reg
        return loss

    def get_loss(self, loss_function):
        Loss_Functions = {
            'CEL': nn.CrossEntropyLoss(), 
        }
        return Loss_Functions[loss_function]
    
    def get_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            return None
        schedulers = {
            'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.00001)
        }
        return schedulers[scheduler]

    def initialize_weights(self, layer_init=None, initialisation='Normal'):
        init_methods = {
            'Xavier': init.xavier_uniform_, 
            'Uniform': lambda x: init.uniform_(x, a=-0.1, b=0.1), 
            'Normal': lambda x: init.normal_(x, mean=0, std=0.01), 
            'He': lambda x: init.kaiming_normal_(x, mode='fan_in', nonlinearity='relu')
        }

        self._init_weights = init_methods[initialisation]

        if layer_init is None:
            print('no layer specified')
            parameters = self.named_parameters()
            print(f'{initialisation} initialization for all weights')
        else: 
            parameters = layer_init.named_parameters()
            print(f'{initialisation} initialization for {layer_init}')

        for name, param in parameters:
            if 'weight' in name:
                self._init_weights(param)
            elif 'bias' in name:
                # Initialize biases to zeros
                nn.init.constant_(param, 0)

