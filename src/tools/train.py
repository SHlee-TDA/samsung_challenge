from typing import Optional, Union
import datetime
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, OneCycleLR
from tqdm import tqdm
from src.tools.metrics import compute_mIoU
from src.visualization.plotting import monitor_training_process






class TrainingConfig:
    """
    A configuration and utility class for training a DANN model.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 weight_decay: float = 0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose: bool = True,
                 print_every: int = 10,
                 criterion: torch.nn.Module = nn.CrossEntropyLoss(),
                 optimizer_choice: str = 'Adam',
                 scheduler_choice: Optional[str] = None,
                 optimizer_params: Optional[dict] = None,
                 scheduler_params: Optional[dict] = None,
                 early_stopping_patience: int = 10,
                freeze_bn: bool = True
):
        """
        Initialize the training configuration.

        Args:
            model (torch.nn.Module): The model to be trained.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
            epochs (int, optional): Number of epochs for training. Default is 1000.
            batch_size (int, optional): Batch size for the dataloaders. Default is 32.
            weight_decay (float, optional): Weight decay for the optimizer. Default is 0.
            device (str, optional): Device for training ("cuda" or "cpu"). Default is "cuda" if available, otherwise "cpu".
            verbose (bool, optional): Whether to print training progress. Default is True.
            print_every (int, optional): How often to print training progress. Default is 10.
            optimizer_choice (str, optional): Choice of optimizer ("Adam" or "SGD"). Default is "Adam".
            scheduler_choice (str, optional): Choice of learning rate scheduler. Default is None.
            optimizer_params (dict, optional): Additional parameters for the optimizer. Default is None.
            scheduler_params (dict, optional): Additional parameters for the scheduler. Default is None.
            early_stopping_patience (int, optional): Number of epochs to wait before early stopping. Default is 10.
        """
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
        #     multi_gpu_train = True
        # elif torch.cuda.device_count() == 1:
        #     print(f"Using only 1 GPU!")
        #     model.to(device)
        #     multi_gpu_train = False
        # else:
        #     print(f"Using CPU")
        #     model.to(device)
        #     multi_gpu_train = False
            
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.print_every = print_every
        self.criterion = criterion
        self.optimizer_choice = optimizer_choice
        self.scheduler_choice = scheduler_choice
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}
        self.early_stopping_patience = early_stopping_patience
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            self.model.freeze_bn()

        # Set the checkpoint directory
        self.model_name = type(model).__name__
        self.start_time = datetime.datetime.now().strftime('%y%m%d_%H%M')
        self.checkpoint_dir = os.path.join(self.start_time + '_' + self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_{self.start_time}.pth")

    def train_DANN(self, source_dataloader, target_dataloader, val_dataloader=None, save_checkpoint_every=None):
            """
            Train a DANN model using the provided data and configuration.

            Args:
                source_dataloader (DataLoader): DataLoader for the source domain data.
                target_dataloader (DataLoader): DataLoader for the target domain data.
                val_dataloader (DataLoader, optional): DataLoader for validation data.
                save_checkpoint_every (int, optional): Epoch interval to save checkpoints. If None, checkpoints are not saved.

            Returns:
                nn.Module: Trained DANN model.
            """



            source_iter = iter(source_dataloader)
            target_iter = iter(target_dataloader)
            
            #optimizer = self._initialize_optimizer(self.model.parameters())
            optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
            # Learning Rate Scheduler Initialization (if provided)
            scheduler = None
            if self.scheduler_choice is not None:
                scheduler = self._initialize_scheduler(optimizer)


            best_mIoU = 0.0

            for epoch in tqdm(range(self.epochs), desc="Training"):
                self.model.train()
                total_loss = 0
                total_domain_loss = 0
                total_semantic_loss = 0
                
                for _ in range(max(len(source_dataloader), len(target_dataloader))):
                    # Source dataloader에서 데이터 가져오기
                    try:
                        source_data, source_labels = next(source_iter)
                        source_data, source_labels = source_data.float().to(self.device), source_labels.long().to(self.device)
                    except StopIteration:
                        source_iter = iter(source_dataloader)
                        source_data, source_labels = next(source_iter)
                        source_data, source_labels = source_data.float().to(self.device), source_labels.long().to(self.device)
                    
                    # Target dataloader에서 데이터 가져오기
                    try:
                        target_data = next(target_iter)
                        target_data = target_data.float().to(self.device)
                    except StopIteration:
                        target_iter = iter(target_dataloader)
                        target_data = next(target_iter)
                        target_data = target_data.float().to(self.device)

                    # Training step
                    semantic_loss, domain_loss, loss = self._train_step(self.model, source_data, source_labels, target_data, optimizer)
                    total_loss += loss.item()
                    total_domain_loss += domain_loss.item()
                    total_semantic_loss += semantic_loss.item()

                # Print training stats
                if self.verbose and (epoch+1) % self.print_every == 0:
                    avg_loss = total_loss / len(source_dataloader)
                    avg_domain_loss = total_domain_loss / len(source_dataloader)
                    avg_semantic_loss = total_semantic_loss / len(source_dataloader)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Average Loss: {avg_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}, Semantic Loss: {avg_semantic_loss:.4f}")

                    # Compute mIoU for source domain training data and validation data (if provided)
                    train_mIoU = self._evaluate_segmentation_mIoU(self.model, source_dataloader)
                    print(f"Epoch [{epoch+1}/{self.epochs}],Training mIoU: {train_mIoU:.4f}")

                    # Validation and early stopping
                    if val_dataloader:
                        avg_val_loss, val_mIoU = self._valid_step(self.model, val_dataloader)
                        print(f"Epoch [{epoch+1}/{self.epochs}], Valid Loss: {avg_val_loss:.4f},Validation mIoU: {val_mIoU:.4f}")
                        
                        # Save checkpoint if the current model has better performance
                        if val_mIoU > best_mIoU:
                            best_mIoU = val_mIoU
                            if save_checkpoint_every and (epoch + 1) % save_checkpoint_every == 0:
                                self._save_checkpoint(self.model, epoch, best_mIoU, self.checpoint_path)


                    # Display segmentation results on training data
                    monitor_training_process(source_dataloader, self.model, self.device, is_domain_classification=False)

                    # Display domain classification results on target data
                    monitor_training_process(target_dataloader, self.model, self.device, is_domain_classification=True)

                # Update the learning rate if the scheduler is provided
                if scheduler:
                    scheduler.step()

            return model

    def _train_step(self, model, source_data, source_labels, target_data, optimizer):
        # Forward pass for source domain
        model.train()
        src_semantic_outputs, src_domain_outputs = model(source_data)
        src_semantic_loss = self.criterion(src_semantic_outputs, source_labels)
        src_domain_loss = F.binary_cross_entropy_with_logits(src_domain_outputs, torch.zeros_like(src_domain_outputs))

        # Forward pass for target domain
        _, tgt_domain_outputs = model(target_data, lamda=-1)  # Set lamda=-1 for gradient reversal
        tgt_domain_loss = F.binary_cross_entropy_with_logits(tgt_domain_outputs, torch.ones_like(tgt_domain_outputs))

        # Combine losses
        domain_loss = src_domain_loss + tgt_domain_loss
        loss = src_semantic_loss + domain_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return src_semantic_loss, domain_loss, loss

    def _valid_step(self, model, val_dataloader):
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_data, val_labels in val_dataloader:
                val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                val_outputs, _ = model(val_data)
                val_loss = nn.CrossEntropyLoss()(val_outputs['out'], val_labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader) if val_dataloader else 0.0
        # Compute mIoU
        _, predictions = torch.max(val_outputs, 1)
        mIoU = self._evaluate_segmentation_mIoU(model, val_dataloader)

        return avg_val_loss, mIoU

    def _evaluate_segmentation_mIoU(self, model, dataloader):
        model.eval()
        total_mIoU = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = model(images)
                _, preds = torch.max(outputs, 1)
                mIoU = compute_mIoU(preds, labels)
                total_mIoU += mIoU

        return total_mIoU / len(dataloader)
    
    def _save_checkpoint(self, model, epoch, best_mIoU, filename):
        """
        Save the model checkpoint.

        Args:
            model (nn.Module): The DANN model to save.
            epoch (int): Current epoch.
            best_mIoU (float): Best mIoU score so far.
            filename (str): Name of the file to save the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_mIoU': best_mIoU
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename} at epoch {epoch} with mIoU {best_mIoU:.4f}.")


    def load_checkpoint(self, model, filename):
        """
        Load the model checkpoint.

        Args:
            model (nn.Module): The DANN model to load.
            filename (str): Name of the checkpoint file to load.

        Returns:
            epoch (int): Epoch of the loaded checkpoint.
            best_mIoU (float): Best mIoU of the loaded checkpoint.
        """
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filename} at epoch {checkpoint['epoch']} with mIoU {checkpoint['best_mIoU']:.4f}.")
        return checkpoint['epoch'], checkpoint['best_mIoU']
    
    def _initialize_optimizer(self, parameters) -> torch.optim.Optimizer:
        """
        Initialize the optimizer based on user's choice and parameters.

        Args:
            parameters: Parameters of the model.

        Returns:
            Initialized optimizer.
        """
        optimizer_choices = {
            'Adam': Adam(filter(lambda p: p.requires_grad, parameters), lr=self.learning_rate, weight_decay=self.weight_decay, **self.optimizer_params),
            'AdamW': AdamW(filter(lambda p: p.requires_grad, parameters), lr=self.learning_rate, weight_decay=self.weight_decay, **self.optimizer_params),
            'SGD': SGD(filter(lambda p: p.requires_grad, parameters), lr=self.learning_rate, weight_decay=self.weight_decay, **self.optimizer_params)
        }
        return optimizer_choices[self.optimizer_choice]

    def _initialize_scheduler(self, optimizer):
        scheduler_choices = {
            'StepLR': StepLR(optimizer, step_size=200, gamma=0.5),
            'ExponentialLR': ExponentialLR(optimizer, gamma=0.95),
            'OneCycleLR': OneCycleLR(optimizer, max_lr=1e-2, total_steps=self.epochs * self.batch_size, 
                       div_factor=25, pct_start=0.3, anneal_strategy='cos')
        }
        return scheduler_choices[self.scheduler_choice]

    