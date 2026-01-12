import numpy as np

# --- IMPORTS FROM OUR  MODULES ---from 
from model.optimizers import Optimizer
from model.mlp import MLP
from model.loss import loss
from model.scheduler import LearningRateScheduler
from data_pipeline.data_generator import DataGenerator
from utils.logger import LoggerSetup  

class NeuralNetworkPipeline:
    def __init__(self, layer_sizes,activations=None , lr=0.1 ,batch_size=32, l_lambda = 0.0001,regularization ='L2',loss_type='BCE',dropout_rates=None ,optimizer_method='adam', 
                 beta1=0.9, beta2=0.999, epsilon=1e-8,lr_scheduler='constant', **scheduler_kwargs): 
        

        # --- 1. SETUP LOGGER ---
        self.logger = LoggerSetup.setup_logger()
        self.logger.info("Initializing Neural Network Pipeline...")

        # Initialize Optimizer
        self.optimizer = Optimizer(
            method=optimizer_method, 
            lr=lr, 
            beta1=beta1, 
            beta2=beta2, 
            epsilon=epsilon
        )

        # Initialize MLP (The Brain)
        self.mlp = MLP(layer_sizes,activations,l_lambda,regularization,dropout_rates,optimizer=self.optimizer)

        # Store Hyperparameters
        self.initial_lr = lr
        self.batch_size = batch_size
        self.l_lambda = l_lambda
        self.regularization = regularization
        self.loss_type = loss_type
        self.dropout_rates = dropout_rates

        # Scheduler Setup
        if hasattr(LearningRateScheduler,lr_scheduler):
            self.scheduler_func = getattr(LearningRateScheduler,lr_scheduler)
            self.logger.debug(f"Scheduler set to: {lr_scheduler}") # <--- DEBUG LOG
        else:
            self.logger.warning(f"Scheduler '{lr_scheduler}' not found. Using constant.") # <--- WARNING LOG
            self.scheduler_func = LearningRateScheduler.constant
        
        self.scheduler_kwargs = scheduler_kwargs 

        # History Storage
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.lr_history = []
    def accuracy(self, y_pred, y_true):
        # Case A: Binary (Output size 1)
        if y_pred.shape[1] == 1:
            predictions = (y_pred >= 0.5).astype(int)
            return np.mean(predictions == y_true) * 100
            
        # Case B: Multi-Class (Output size > 1)
        else:
            # 1. Get Predicted Class (Index of max value)
            pred_class = np.argmax(y_pred, axis=1)
            
            # 2. Get True Class
            # If y_true is One-Hot (N, 10), convert to Index
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                true_class = np.argmax(y_true, axis=1)
            else:
                true_class = y_true.flatten()
                
            return np.mean(pred_class == true_class) * 100

    #-------------------------------------
    #         Regularization
    #---------------------------------------
    def calculate_data_loss(self,y_pre,y_true) : 
        self.size = y_true.shape[0]
        if self.loss_type == 'MSE':
            return loss.MSE(y_pre, y_true)
        elif self.loss_type == 'BCE':
            return loss.BCE(y_pre, y_true)
        
    def regularization_loss(self, data_loss):
        if self.regularization not in ['L2', 'L1']:
            return data_loss
        
        regularization_term = 0.0 

        for layer in self.mlp.layers:
            if self.regularization == 'L2':
                regularization_term += np.sum(layer.w ** 2)
            elif self.regularization == 'L1':
                regularization_term += np.sum(np.abs(layer.w))

        if self.regularization == 'L2':
            penalty = (self.l_lambda / (2 * self.size)) * regularization_term
        elif self.regularization == 'L1':
            penalty = (self.l_lambda / self.size) * regularization_term

        return data_loss + penalty
    
    def compute_total_loss(self, y_pre, y_true):
        data_loss = self.calculate_data_loss(y_pre, y_true)
        total_loss = self.regularization_loss(data_loss)
        return total_loss
    
    #------------------------------------------------------------
    # Backward Pass
    #-------------------------------------------------------------
    def total_backward(self,y_pred,y_true): 
       if  self.loss_type == 'MSE':
           return loss.MSE_backward(y_pred,y_true)
       elif self.loss_type == 'BCE':
           return loss.BCE_backward(y_pred,y_true)

    #------------------------------------------------------------
    # Training Loop (Refactored with DataGenerator)
    #-------------------------------------------------------------
    def train(self,x_train,y_train,x_val=None,y_val=None,epochs=1000,verbose=True):
        n_samples = x_train.shape[0]
        t = 0 

        # REFACTOR: Using DataGenerator instead of manual loop
        train_gen = DataGenerator(x_train, y_train, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            
            # A. Update Learning Rate
            self.scheduler_kwargs['total_epochs'] = epochs 
            current_lr = self.scheduler_func(
                self.initial_lr, 
                epoch,
                **self.scheduler_kwargs 
            )

            # B. Update Optimizer
            self.optimizer.lr = current_lr
            self.lr_history.append(current_lr) 

            # C. Batch Loop (Using Generator)
            for x_batch, y_batch in train_gen:
                t += 1
                
                # Forward
                y_pred = self.mlp.forward(x_batch, traning=True) 
                
                # Backward
                dl_doutput = self.total_backward(y_pred, y_batch)
                gradients = self.mlp.backward(dl_doutput)
                
                # Update
                self.mlp.update_parameters(gradients, t)


            # D. Metrics (Epoch End)
            y_pred_train = self.mlp.forward(x_train, traning=False)
            train_loss = self.compute_total_loss(y_pred_train, y_train) 
            train_accuracy = self.accuracy(y_pred_train, y_train)

            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_accuracy)

            val_info = ""
            # Validation
            if x_val is not None and y_val is not None:
                y_pred_val = self.mlp.forward(x_val, traning=False)
                val_loss = self.compute_total_loss(y_pred_val, y_val)
                val_accuracy = self.accuracy(y_pred_val, y_val)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)
                val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
            
            # Logging
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | Acc: {train_accuracy:.2f}%{val_info}"
                self.logger.info(msg) # <--- INFO LOG


    def predict_proba(self, x):
        """Returns raw probabilities/activations"""
        return self.mlp.forward(x, traning=False)

    def predict(self, x):
        """Returns Class Indices (0, 1 for Binary | 0-9 for MNIST)"""
        activations = self.predict_proba(x)
        
        # Case A: Binary (Output size 1)
        if activations.shape[1] == 1:
            return (activations >= 0.5).astype(int)
        
        # Case B: Multi-Class (Output size > 1) -> Pick highest prob
        else:
            # Returns indices like [5, 0, 9, ...]
            return np.argmax(activations, axis=1).reshape(-1, 1)


                        






                                                


                        
