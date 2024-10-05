import os
import argparse
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from CNNclassifier import CNNThai
from Trainer import Trainer
from CustomDataClass import create_dataframe, create_label_dictionary, DataModule
import matplotlib.pyplot as plt
import pickle


# Set the environment variable to make the specified GPU visible

parser = argparse.ArgumentParser(description="Iterate through a directory and list files and folders and train a CNN on the image data.")

parser.add_argument("directory", type=str, help="Path to the directory")
parser.add_argument("language", type=str, help="Determine Langauge")
parser.add_argument("--resolution", type=str, default = '200', help="Set resolution")
parser.add_argument("--style", type=str, default = 'normal', help="Set style")
parser.add_argument("--add_language", type=str, help="Determine additional Langauge")
parser.add_argument("--hyper", help="Use Hyperparameter optimization", action="store_true" )
parser.add_argument('--cuda_device', type=int, default=0, help='Specify the CUDA device number (default: 0)')


args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

hyperparameter_optimization = args.hyper
if hyperparameter_optimization == True:
  print("Hyperparameter optimization:", hyperparameter_optimization)

#################################################################  
folder = args.directory
language = args.language
directory_path = folder + '/' + language

resolution = args.resolution
style = args.style
image_specifics = resolution + '/' + style

#Allow the sets to be constrained by choice of language (Thai, English, or both). 
df = create_dataframe(directory_path, image_specifics)

if args.add_language is not None:
    directory_path_2 = folder + '/' + args.add_language
    df2 = create_dataframe(directory_path_2, image_specifics)
    df = pd.concat([df, df2], axis=0, ignore_index=True)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

label_dictionary = create_label_dictionary(train_df.labels)
train_data = DataModule(train_df.images, train_df.labels.replace(label_dictionary))
val_data = DataModule(val_df.images, val_df.labels.replace(label_dictionary))
test_data = DataModule(test_df.images, test_df.labels.replace(label_dictionary))
#################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_of_labels = len(label_dictionary)

#Train the model with an option of hyperparameter optimization
if hyperparameter_optimization == True:
    print('Hyperparameter Optimization Loop')
    best_params, best_accuracy = Trainer.hyperparameter_optimization(train_data, val_data,  num_of_labels, device, n_trials = 15)
    cnn_model= CNNThai(best_params['hidden_size'], num_of_labels, optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate']).to(device)
    trainer = Trainer(50, best_params['batch_size'], early_stopping_patience = 7)
    trainer.fit(cnn_model, train_data, val_data)

else: 
    cnn_model = CNNThai(64, output_size=num_of_labels, optimizer = 'SGD', learning_rate = 0.003388, l2 = 0.0001).to(device)
    trainer = Trainer(max_epochs = 4, batch_size = 16)
    trainer.fit(cnn_model,train_data,val_data)

#Plot Progress report
n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss,nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss,nan_values])

plt.figure(figsize=(10,6))
plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
plt.title("Train Loss")
plt.legend()

#################################################################

#################################################################
#Test the model
trainer.test(cnn_model, test_data)

torch.save(cnn_model.state_dict(), 'model_parameters.pth')
torch.save(cnn_model, 'model.pth')

