import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface

from .datasets import WindowDataset

logger = logging.getLogger(__name__)

class PyTorchLSTMTransformerTrainer(PyTorchLSTMModelTrainer):
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
