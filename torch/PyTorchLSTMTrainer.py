import logging
from pathlib import Path
from typing import Any, Tuple, Optional

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface
from freqtrade.freqai.torch.PyTorchLSTMModelTrainer import PyTorchLSTMModelTrainer

from .datasets import WindowDataset


logger = logging.getLogger(__name__)


class PyTorchLSTMTrainer(PyTorchLSTMModelTrainer):
    """
    Creating a trainer for the LSTM model.
    """
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
                                 
            device: str,
            data_convertor: PyTorchDataConvertor,
            criterion: nn.Module = None,
            model_meta_data: dict[str, Any] = {},
            window_size: int = 1,
            tb_logger: Any = None,
            **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            data_convertor=data_convertor,
            criterion=criterion,
            model_meta_data=model_meta_data,
            window_size=window_size,
            tb_logger=tb_logger,
            **kwargs,
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001
        )

    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]):
        self.model.train()

        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary["train_features"])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter = 0
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _, batch_data in enumerate(data_loaders_dictionary["train"]):
                                                                     
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                # loss = self.criterion(yb_pred.squeeze(), yb.squeeze())
                loss = self.custom_loss_function(yb_pred.squeeze(), yb.squeeze())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                       
                        
                self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1
                epoch_loss += loss.item()

            # evaluation
            if "test" in splits:
                test_loss = self.estimate_loss(data_loaders_dictionary, "test")
                self.learning_rate_scheduler.step(test_loss)  # Update the learning rate scheduler

            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {epoch_loss / len(data_loaders_dictionary['train']):.4f}")

    def create_data_loaders_dictionary(
            self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        Uses WindowDataset to create windows of data for LSTM.
        """
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            data_loader = DataLoader(
                dataset,
                ## 防止ZeroDivisionError
                # batch_size=self.batch_size,
                # shuffle=True,
                # drop_last=True,
                batch_size=len(dataset),
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary

    @torch.no_grad()
    def estimate_loss(
            self,
            data_loader_dictionary: dict[str, DataLoader],
            split: str,
    ) -> float:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        for _, batch_data in enumerate(data_loader_dictionary[split]):
            xb, yb = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            yb_pred = self.model(xb)
            # loss = self.criterion(yb_pred.squeeze(), yb.squeeze())
            loss = self.custom_loss_function(yb_pred.squeeze(), yb.squeeze())
            total_loss += loss.item()
            num_batches += 1
            self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1

        self.model.train()
                            
                                                                                                                      
                    
             
        return total_loss / num_batches
