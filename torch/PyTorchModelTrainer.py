import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface

from .datasets import WindowDataset

logger = logging.getLogger(__name__)


class PyTorchModelTrainer(PyTorchTrainerInterface):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str,
        data_convertor: PyTorchDataConvertor,
        criterion: nn.Module = None,  # 将 criterion 设置为可选参数
        model_meta_data: dict[str, Any] = {},
        window_size: int = 1,
        tb_logger: Any = None,
        **kwargs,
    ):
        """
        :param model: The PyTorch model to be trained.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        :param init_model: A dictionary containing the initial model/optimizer
            state_dict and model_meta_data saved by self.save() method.
        :param model_meta_data: Additional metadata about the model (optional).
        :param data_convertor: converter from pd.DataFrame to torch.tensor.
        :param n_steps: used to calculate n_epochs. The number of training iterations to run.
            iteration here refers to the number of times optimizer.step() is called.
            ignored if n_epochs is set.
        :param n_epochs: The maximum number batches to use for evaluation.
        :param batch_size: The size of the batches to use during training.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.n_epochs: int | None = kwargs.get("n_epochs", 10)
        self.n_steps: int | None = kwargs.get("n_steps", None)
        if self.n_steps is None and not self.n_epochs:
            raise Exception("Either `n_steps` or `n_epochs` should be set.")

        self.batch_size: int = kwargs.get("batch_size", 64)
        self.data_convertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter = 0

    def custom_loss_function(self, y_pred, y_true):
        # 原始MSE损失
        return self.criterion(y_pred, y_true)
    
        # # 自定义损失函数
        # # 基础MSE损失
        # base_loss = (y_pred - y_true) ** 2
        # # 非零值附近的权重
        # nonzero_weights = torch.where(y_true != 0, 5.0, 0.5)
        # # 大误差惩罚
        # large_error_penalty = torch.where(
        #     torch.abs(y_pred - y_true) > 0.5,
        #     torch.abs(y_pred - y_true) ** 3,  # 更强的惩罚（使用立方而不是平方）
        #     0.0
        # )
        # # 对突增后的回归处理的惩罚
        # recovery_penalty = torch.where(
        #     (y_true == 0) & (torch.abs(y_pred) > 0.5),  # 如果预测值偏离0太远
        #     torch.abs(y_pred) ** 2,  # 增加更强的惩罚
        #     0.0
        # )
        # # 方向惩罚（鼓励突增/突降幅度匹配）
        # direction_penalty = torch.where(
        #     ((y_true > 0) & (y_pred < y_true)) | ((y_true < 0) & (y_pred > y_true)),
        #     torch.abs(y_pred - y_true) * 5.0,  # 增大方向性误差惩罚
        #     0.0
        # )
        # # 额外惩罚：鼓励模型输出的突增幅度尽量接近真实值
        # amplitude_penalty = torch.where(
        #     torch.abs(y_true - y_pred) > 0.5,  # 只关注较大的误差
        #     torch.abs(y_true - y_pred) ** 2 * 2.0,  # 增强幅度差距的惩罚
        #     0.0
        # )
        # # 组合所有损失项
        # total_loss = (base_loss * nonzero_weights) + large_error_penalty + direction_penalty + recovery_penalty + amplitude_penalty
        # return total_loss.mean()


    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]):
        """
        :param data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        :param splits: splits to use in training, splits must contain "train",
        optional "test" could be added by setting freqai.data_split_parameters.test_size > 0
        in the config file.

         - Calculates the predicted output for the batch using the PyTorch model.
         - Calculates the loss between the predicted and actual output using a loss function.
         - Computes the gradients of the loss with respect to the model's parameters using
           backpropagation.
         - Updates the model's parameters using an optimizer.
        """
        self.model.train()

        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary["train_features"])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter = 0
        for _ in range(n_epochs):
            for _, batch_data in enumerate(data_loaders_dictionary["train"]):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                # loss = self.criterion(yb_pred.squeeze(), yb.squeeze())
                # 使用自定义损失函数
                loss = self.custom_loss_function(yb_pred.squeeze(), yb.squeeze())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1

            # evaluation
            if "test" in splits:
                self.estimate_loss(data_loaders_dictionary, "test")

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
            # 使用自定义损失函数
            # loss = self.criterion(yb_pred.squeeze(), yb.squeeze())
            loss = self.custom_loss_function(yb_pred.squeeze(), yb.squeeze())
            total_loss += loss.item()
            num_batches += 1
            self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1

        self.model.train()
        return total_loss / num_batches

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        """
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = TensorDataset(x, y)
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

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        Calculates the number of epochs required to reach the maximum number
        of iterations specified in the model training parameters.

        the motivation here is that `n_steps` is easier to optimize and keep stable,
        across different n_obs - the number of data points.
        """
        assert isinstance(self.n_steps, int), "Either `n_steps` or `n_epochs` should be set."
        n_batches = n_obs // self.batch_size
        n_epochs = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(
                f"Setting low n_epochs: {n_epochs}. "
                f"Please consider increasing `n_steps` hyper-parameter."
            )

        return n_epochs

    def save(self, path: Path):
        """
        - Saving any nn.Module state_dict
        - Saving model_meta_data, this dict should contain any additional data that the
          user needs to store. e.g. class_names for classification models.
        """

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_meta_data": self.model_meta_data,
                "pytrainer": self,
            },
            path,
        )

    def load(self, path: Path):
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: dict):
        """
        when using continual_learning, DataDrawer will load the dictionary
        (containing state dicts and model_meta_data) by calling torch.load(path).
        you can access this dict from any class that inherits IFreqaiModel by calling
        get_init_model method.
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint["model_meta_data"]
        return self


class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Creating a trainer for the Transformer model.
    """

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
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
                # shuffle=False,
                # drop_last=True,
                batch_size=len(dataset),
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary


class PyTorchLSTMTrainer(PyTorchModelTrainer):
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