import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from CustomDataClass import create_dataframe, create_label_dictionary, DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from CNNclassifier import CNNThai
from Trainer import Trainer
import os
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Iterate through a directory and list files and folders.")
parser.add_argument("directory", type=str, help="Path to the directory")
parser.add_argument('--cuda_device', type=int, default=0, help='Specify the CUDA device number (default: 0)')




args = parser.parse_args()
directory = args.directory
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)


Experiments =[[['Thai', 'normal', '200'],['Thai','normal','200']],
                [['Thai', 'normal', '400'],['Thai','normal','200']],
                [[['Thai', 'English'], 'normal', '200'],[['Thai', 'English'],'normal','200']],
                [['Thai', 'normal', '400'],['Thai','bold','400']], 
                [['Thai', 'bold', '200'],['Thai','normal','200']],
                [['Thai', ['normal','bold','bold_italic','italic'], '200'],['Thai',['normal','bold','bold_italic','italic'],'200']],
                [[['Thai', 'English'], ['normal','bold','bold_italic','italic'], '200'],[['Thai', 'English'],['normal','bold','bold_italic','italic'],'200']]]



################################################################
#The following three functions are used to create the corresponding datasets for the experiments
################################################################

def create_lists(experiment):
    list_of_lists = []
    for item in experiment: 
        if isinstance(item, list):
            list_of_lists.append(item)
        else:
            list_of_lists.append([item])
    combinations = list(itertools.product(*list_of_lists))
    return combinations

def create_dataset(experiment, directory):
    dataset = pd.DataFrame(columns=['images', 'labels'])
    for item in experiment: 
        folder = directory + '/' + item[0]
        directory_path = item[2] + "/" + item[1] 
        df = create_dataframe(folder, directory_path)
        dataset = pd.concat([dataset,df], axis=0, join='outer', ignore_index=True)
    return dataset

def get_datasets(experiment):
    if experiment[0] == experiment[1]:
        train_test_set = create_lists(experiment[0])
        dataset = create_dataset(train_test_set, directory)
        train_df, temp_df = train_test_split(dataset, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        label_dictionary = create_label_dictionary(train_df.labels)
        train_data = DataModule(train_df.images, train_df.labels.replace(label_dictionary))
        val_data = DataModule(val_df.images, val_df.labels.replace(label_dictionary))
        test_data = DataModule(test_df.images, test_df.labels.replace(label_dictionary))

    else:
        training_set = create_lists(experiment[0])
        testing_set = create_lists(experiment[1])
        train_val_set = create_dataset(training_set, directory)
        test_set = create_dataset(testing_set, directory)
        train_df, val_df = train_test_split(train_val_set, test_size=0.4, random_state=42)
        test_df = test_set.sample(frac=1).reset_index(drop=True)
        
        label_dictionary = create_label_dictionary(train_df.labels)
        train_data = DataModule(train_df.images, train_df.labels.replace(label_dictionary))
        val_data = DataModule(val_df.images, val_df.labels.replace(label_dictionary))
        test_data = DataModule(test_df.images, test_df.labels.replace(label_dictionary))
    
    return train_data, val_data, test_data, label_dictionary

######################################################################

######################################################################
#Plot the training and validation loss, if desired
######################################################################

def plot_progress(trainer):
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
#Iterate through the list of experiments, train and test the model on the respective sets
#Write the results in a .txt. file
######################################################################

total_exp = len(Experiments)
exp_no = 1
for experiment in Experiments: 
    print(f'Experiment number: {exp_no}/{total_exp}')
    train_data, val_data, test_data, label_dictionary = get_datasets(experiment)
    num_of_labels = len(label_dictionary)
    cnn_model = CNNThai(64, num_of_labels, optimizer = 'SGD', learning_rate = 0.00338, l2 = 0.0001, scheduler = 'OnPlateau').to(device)
    cnn_model = CNNThai(output_size=num_of_labels, learning_rate = 0.003388, ).to(device)
    trainer = Trainer(max_epochs = 1, batch_size = 16)
    trainer.fit(cnn_model,train_data,val_data)
    #plot_progress(trainer)
    #plt.show()

    precision, recall, f1, accuracy = trainer.test(cnn_model, test_data)
    

    with open('experiments.txt', mode='a') as file:
        line = f"Experiment number: {exp_no}. "
        file.write(line)
        line = "Training_Set: " 
        file.write(line)
        for item in experiment[0]:
            if isinstance(item, str):
                file.write(item +' ; ')
            else: 
                file.write(", ".join(item) + " ; ")
        line = "Testing_Set: " 
        file.write(line)
        for item in experiment[1]:
            if isinstance(item, str):
                file.write(item +' ; ')
            else: 
                file.write(", ".join(item) + " ; ")
        line = "\n"
        file.write(line)
        line = f"Precision: {precision:.2f} "
        file.write(line)
        line = f"Recall: {recall:.2f} "
        file.write(line)
        line = f"F1 Score: {f1:.2f} "
        file.write(line)
        line = f"Accuracy: {accuracy:.2f} "
        file.write(line)
        file.write('\n\n')
    exp_no += 1
