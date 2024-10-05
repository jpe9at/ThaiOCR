import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from CNNclassifier import CNNThai


class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs = 30, batch_size = 8, early_stopping_patience=6, min_delta = 0.0007):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.num_epochs_no_improve = 0
        self.min_delta = min_delta
    

    def prepare_training_data(self, data_train, data_val, batch_size):
        self.train_dataloader = data_train.train_dataloader(batch_size)
        self.val_dataloader = data_val.val_dataloader(batch_size)
    
    def prepare_testing_data(self, data_test, batch_size):
        self.test_dataloader = data_test.get_dataloader(batch_size)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, data_train, data_val):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(data_train, data_val, self.batch_size)
        self.prepare_model(model)
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, val_loss = self.fit_epoch()
            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss: .4f}, LR = {self.model.scheduler.get_last_lr() if self.model.scheduler is not None else self.model.learning_rate}')
            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            #########################################
            #Early Stopping Monitor
            #instead, we can also use the early stopping monitor class below. 
            if (self.best_val_loss - val_loss) > self.min_delta:
                self.best_val_loss = val_loss
                self.num_epochs_no_improve = 0
            else:
                self.num_epochs_no_improve += 1
                if self.num_epochs_no_improve == self.early_stopping_patience:
                    print("Early stopping at epoch", epoch)
                    break
            ########################################

            ########################################
            #Scheduler for adaptive learning rate
            if self.model.scheduler is not None:
                self.model.scheduler.step(val_loss)
            ########################################


    def fit_epoch(self):
        train_loss = 0.0
        for x_batch, y_batch in self.train_dataloader:
            print('allocated_device')
            print(next(self.model.parameters()).device)
            #x_batch = x_batch.to(self.model.device)
            y_batch = y_batch.to(next(self.model.parameters()).device)
            output = self.model(x_batch)
            print(output.device)
            loss = self.model.loss(output, y_batch)
            self.model.optimizer.zero_grad()
            loss.backward()
            
            ######################################
            #L1 Loss
            if self.model.l1_rate != 0: 
                loss = self.model.l1_regularization(self.model.l2_rate)
            ######################################
            
            self.model.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(self.train_dataloader.dataset)
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.val_dataloader:
                val_output = self.model(x_batch)
                y_batch = y_batch.to(next(self.model.parameters()).device)
                loss = self.model.loss(val_output, y_batch)
                val_loss += loss.item() * x_batch.size(0) #why multiplication with 0?
            val_loss /= len(self.val_dataloader.dataset)
        return train_loss, val_loss

    def test(self, model, data_test):
        model.eval()
        self.test_dataloader = data_test.get_dataloader(self.batch_size)
        y_pred = []
        y = []
        with torch.no_grad():
            for X_batch,y_batch in self.test_dataloader:
                y_batch = y_batch.to(next(self.model.parameters()).device)
                y_hat = torch.argmax(model(X_batch), dim=1)  # Choose the class with highest logits
                y_pred.append(y_hat)
                y.append(y_batch)
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)

        precision = precision_score(y, y_pred, average='macro')  # or 'micro', 'weighted'
        recall = recall_score(y, y_pred, average='macro')        # or 'micro', 'weighted'
        f1 = f1_score(y, y_pred, average='macro')                # or 'micro', 'weighted'
        accuracy = accuracy_score(y, y_pred)

        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        
        return precision, recall, f1, accuracy 

#    def calculate_accuracy(self,predictions, labels):
#        # Get the predicted classes by selecting the index with the highest probability
#        #predicted_classes = torch.argmax(predictions, 0)
#        # Compare predictions with ground truth
#        correct_predictions = torch.eq(predictions, labels).sum().item()
#        # Calculate accuracy
#        accuracy = correct_predictions / labels.size(0)
#    
#        return accuracy

    @classmethod
    def Optuna_objective(cls, trial, train_data, val_data, output_size, device):
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        learning_r = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16,32,128])
        hidden_size = trial.suggest_categorical('hidden_size',[64,128])
        l2_rate = trial.suggest_categorical('l2_rate', [0.0,0.0001,0.005])

        model = CNNThai(hidden_size, output_size, optimizer = optimizer, learning_rate = learning_r, l2 = l2_rate, scheduler = 'OnPlateau').to(device)
        trainer = cls(40,  batch_size )
        trainer.fit(model, train_data, val_data)

        return  trainer.val_loss_values[-1]

    @classmethod
    def hyperparameter_optimization(cls, train_data, val_data, output_size, device, n_trials = 20):
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: cls.Optuna_objective(trial, train_data, val_data, output_size, device)
        study.optimize(objective_func, n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value

        return best_params, best_accuracy
