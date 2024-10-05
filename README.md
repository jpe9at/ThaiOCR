The project contains two scripts used for training and experimentation; 

### main.py [-h] [--resolution RESOLUTION] [--style STYLE] [--add_language ADD_LANGUAGE] [--hyper] [--cuda_device CUDA_DEVICE] directory language
      
Parameter options: 
      
--directory: the directory of the dataset
                
--resolution: 200, 300, 400
                
--style: normal, bold, italic, bold_italic
                
--cuda_device: select GPU name. 
                
If cuda_device is called, only the specified GPU will be visible to pytorch. 
                
### experiments.py [-h]  [--cuda_device CUDA_DEVICE] directory 
     
Parameter options: 
      
--directory: the directory of the dataset
                
--cuda_device: select GPU name

If cuda_device is called, only the specified GPU will be visible to pytorch.

### Description 

The set up follows the dive into deep learning book https://d2l.ai/index.html. With a Data Module names CustomDataClass.py to create the training, validation, and test sets, a trainer module Trainer.py, and the convolutional Network CNNClassifier.py. 

main.py is the main scrip calling those modules. It contains an option to run a hyperparameter optimization loop or a simple training with predetermined paramenters. There are also options to determine the resolution, and style of the images to be classified, with the possibility of joinly training on English and Thai characters. 

experiments.py contains a list of experiments with different selections of languages, resolutions, styles for both the training/validation set and the training set, with the possibility to select different options for both. It iterates through the different experimental set-ups and writes precision, recall, F1, and accuracy values to the file experiments.txt.

CNNClassifier.py contains a convolutional NN with the conv layer with 16, 32, and 64 channels each, followed by two fully connected layers with hidden size 64. No Dropout was included in the final version. Default weight initialisation. 

Trainer.py contains the training and testing functions, a learnign rate scheduler, an early stopping monitor, and a method for hyperparameter optimization using optuna. Parameters after optimization on joint English and Thai datasets with normal style are: Optimizer = SGD, learning_rate = 0.0038, l1 = 0.0001, batch_size = 16.

### Some observations from the experiments

The results in the experiments.txt file are quite uniform, with no dramatic outliers. Accuracy is a little higher when training and test data align in style, and a little less when not (which is no surprise). Training on Thai and English jointly has no dramatic negative effect on performance. An interesting observation is that combining multiple styles improves overall performance (with an accuracy of 0.96 for Thai only including all styles). 

Changes in style reduces performance by a small margin, esp. from bold to italic delivers the worst result with 0.86 accuracy. 
