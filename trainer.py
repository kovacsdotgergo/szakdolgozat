import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Epoch_counter():
    def __init__(self, max_batch):
        self.epoch = 0
        self.batch = 0
        self.max_batch = max_batch
        
    def inc_epoch(self):
        self.epoch += 1
    
    def get_epoch(self):
        return self.epoch

    def get_epoch_float(self):
        return self.epoch + self.batch / self.max_batch

    def inc_batch(self):
        self.batch +=1

    def get_batch(self):
        return self.batch

    def reset_batch(self):
        self.batch = 0

class Training_stats_logger():
    def __init__(self, max_batch):
        self.train_losses = []
        self.last_train_loss_count = 0
        self.last_train_loss_sum = 0
        self.epoch_counter = Epoch_counter(max_batch)
        self.last_train_stats = {
                'val_loss': [],
                'val_acc': [],
                'avg_train_loss': [],
                'epoch': []
        }

    def add_train_loss(self, loss):
        self.train_losses.append(loss)
        self.last_train_loss_count += 1
        self.last_train_loss_sum += loss

    def get_avg_loss(self):
        return self.last_train_loss_sum / self.last_train_loss_count \
            if 0 != self.last_train_loss_count else None

    def reset_last_train_loss(self):
        self.last_train_loss_count = 0
        self.last_train_loss_sum = 0

    def save_last_train_stats(self, val_loss, val_acc):
        self.last_train_stats['val_loss'].append(val_loss)
        self.last_train_stats['val_acc'].append(val_acc)
        self.last_train_stats['avg_train_loss'].append(self.get_avg_loss())
        self.last_train_stats['epoch'].append(self.epoch_counter.get_epoch_float())

    def get_last_train_stats(self):
        return self.last_train_stats

    def print_log_message(self, val_loss, val_acc):
        print(f"Avg train loss at {self.epoch_counter.get_epoch()}.epoch, "
            f"{self.epoch_counter.get_batch()}. batch:\t"
            f"{self.get_avg_loss():.3f}\t\t"
            f"Val_loss at {self.epoch_counter.get_epoch()}.epoch, "
            f"{self.epoch_counter.get_batch()}.batch:\t"
            f"{val_loss:.3f}\tacc: {val_acc:.3f}")

class Trainer():
    """@brief class for training and validating models"""
    def __init__(self, model, cuda_available, criterion=nn.CrossEntropyLoss,
                log_message=True):
        """@param[in]   model   deep learning model to train or validate
        @param[in]  criterion   loss function for training and validation
        @param[in]  cuda_available  if cuda is available
        @param[in]  log_message     if log messages during training 
        """
        if cuda_available:
            model = model.cuda()
        self.model = model
        self.criterion = criterion()
        self.cuda_available = cuda_available
        self.last_train_data = None
        self.log_message = log_message 
    
    def train(self, train_loader, val_loader, optimizer=torch.optim.AdamW, lr=0.001,
              train_epochs=30, val_interval=50, save_best_model=True,
              save_path='./tmp.pth'):
        """@brief   trains the model with the given parameters, saves the best model,
            saves the training data for analysis
        @param[in]  train_loader    dataloader for the training data
        @param[in]  val_loader      dataloader for the validation data
        @param[in]  optimizer       optimizer for model training
        @param[in]  lr      learning rate for the optimizer
        @param[in]  train_epochs    duration of training
        @param[in]  val_interval    num of batches after validation is done
        @param[in]  log     bool, if logging is enabled
        @param[in]  save_best_model bool, if saving the best model is required
        @param[in]  save_path       save path to save the best model, should contain the file name as well"""
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        # for saving the best model
        prevloss, _ = self.validate(val_loader)
        minloss = prevloss
        best_model_dict = None
        self.train_stats_logger = Training_stats_logger(len(train_loader))

        for _ in range(train_epochs):
            minloss, best_model_dict = self._step_epoch(train_loader, val_loader, val_interval, minloss,
                    best_model_dict)
        if save_best_model and best_model_dict is not None:
            torch.save(best_model_dict, save_path)       
        if self.log_message:
            print('Finished Training')
    
    def _step_epoch(self, train_loader, val_loader, val_interval, minloss,
                    best_model_dict):
        """@brief   perfoms steps for one epoch, optimizing the model and saving the data
        @param[in]  train_loader    dataloader for the training data
        @param[in]  val_loader      dataloader for the training data
        @param[in]  val_interval    interval in batch to perform validation
        @param[in]  minloss         previous min loss for saving the best model
        @param[in]  best_model_dict dictionary conatining the data to be saved
        @returns    minloss"""
        self.train_stats_logger.reset_last_train_loss()
        self.train_stats_logger.epoch_counter.reset_batch()

        for inputs, labels in train_loader:
            loss = self._step(inputs, labels)
            # summing loss for plotting average
            self.train_stats_logger.add_train_loss(loss)
            self.train_stats_logger.epoch_counter.inc_batch()

            if 0 == self.train_stats_logger.epoch_counter.get_batch() % val_interval:
                # printing statistics
                val_loss, val_accuracy = self.validate(val_loader)

                if self.log_message:
                    self.train_stats_logger.print_log_message(val_loss, val_accuracy)
                self.train_stats_logger.save_last_train_stats(val_loss, val_accuracy)
                self.train_stats_logger.reset_last_train_loss()

                if val_loss < minloss:
                    minloss = val_loss
                    #saving the best model
                    best_model_dict = {
                            'epoch': self.train_stats_logger.epoch_counter.get_epoch_float(),
                            'model': copy.deepcopy(self.model.state_dict()),
                            'optimizer': copy.deepcopy(self.optimizer.state_dict())}
        self.train_stats_logger.epoch_counter.inc_epoch()
        return minloss, best_model_dict
    
    def _step(self, inputs, labels):
        """@brief   performs on training step on the model
        @param[in]  inputs  input data, can be batched
        @param[in]  labels  corresponding labels for calculatin loss
        @returns    loss.item()"""
        if self.cuda_available:
                inputs, labels = inputs.cuda(), labels.cuda()
        self.optimizer.zero_grad()
        self.model.train()

        # forward pass
        outputs = self.model(inputs)
        # calculating loss
        loss = self.criterion(outputs, labels)
        # backpropagation
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, val_loader):
        """@brief   validate model on the dataset wrapped by loader
        @returns    (validation loss, accuracy)
        @param[in]  val_loader  DataLoader class wrapping the val dataset"""
        self.model.eval()
        val_loss = 0
        total = 0
        correct = 0
        
        #calculating the sum of the losses on the validation data
        with torch.no_grad():
            for inputs, labels in val_loader:
                if self.cuda_available:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                #calculating loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                #calculating accuracy
                predictions = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        #returning the mean of the loss on the validation data
        return val_loss/len(val_loader), correct/total
    
    def test(self, test_loader):
        """@brief  tests the model on the test dataset
        @returns   the accuracy on the test set
        @param[in] test_loader DataLoader class wrapping the test dataset"""
        _, test_acc = self.validate(test_loader)        
        return test_acc

    def load_model(self, load_path):
        """TODO"""
        saved_dict = torch.load(load_path)
        epoch = saved_dict['epoch']
        print(f'Loading model from {epoch}.epoch')
        self.model.load_state_dict(saved_dict['model'])

    def inference(self, input, ret_index=True):
        """@brief   caclulates the input for the given input
        @param[in]  input   input tensor
        @param[in]  ret_index   bool, if index of the highest class should be returned
        @returns    the output tensor or highest index if ret_index is true"""
        #for unbatched input the model accepts 3D
        if 2 == input.dim():
            input = input.unsqueeze(dim=0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = self.model.forward(input)
        return torch.argmax(out, dim=1) if ret_index else out

    def plot_train_proc(self, title):
        """@brief   plots validation and running data of the last training
        @param[in]  title   title of the plotted figure"""
        last_train_stats = self.train_stats_logger.get_last_train_stats()
        epochs = last_train_stats['epoch']

        fig, ax_loss = plt.subplots()
        ax_acc = ax_loss.twinx()
        ax_loss.plot(epochs, last_train_stats['val_loss'])
        ax_loss.plot(epochs, last_train_stats['avg_train_loss'])
        ax_acc.plot(epochs, last_train_stats['val_acc'])
        
        ax_loss.set_title(title)
        ax_loss.set_xlabel('epoch')
        ax_loss.set_ylabel('loss', color='g')
        ax_acc.set_ylabel('accuracy', color='b')
        plt.grid()
        plt.show()
                
    def hyperparameter_plotting(self, lrs, train_loader, val_loader,
        optimizer=torch.optim.AdamW, train_epochs=10, val_interval=10,
        log=False):
        """@brief   trains the model for all of the given parameters,
                    then plots the statistics for all of the trainings
        @param[in]  lrs     list or arrays of learning rates
        @param[in]  train_loader    dataloader for training data
        @param[in]  val_loader      dataloader for validation data
        @param[in]  optimizer       optimizer for model training
        @param[in]  train_epochs    number of epochs for each training
        @param[in]  val_interval    num of batches after validation is done
        @param[in]  log             bool, if logging messages should be displayed
        @returns    list of data for each training"""
        hyperparam_data = []
        model_dict = copy.deepcopy(self.model.state_dict())
        for lr in lrs:
            self.train(train_loader, val_loader, optimizer=optimizer, lr=lr,
                train_epochs=train_epochs, val_interval=val_interval, log=log,
                save_best_model=False, log_train_data=True)
            #TODO if append is enough without copy
            hyperparam_data.append((lr, self.last_train_data))
            self.plot_train_proc(f'lr = {lr}')
            self.model.load_state_dict(model_dict)
        return hyperparam_data