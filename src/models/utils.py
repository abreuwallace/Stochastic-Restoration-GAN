import logging
import time
import torch.nn as nn
import torch
from src.models.generator import Generator

def profile_loss(gen, x, v, p_freq=1.3, p_rythm=1.6):
    """Calculate loss profile for prevention of mode collapsing.

    Args:
        gen (Generator): generator model
        x (torch.Tensor): input tensor [B, C, F, T]
        v (float): regularization strength
        p_freq (float): norm order for frequency profile
        p_rythm (float): norm order for rythm profile
    """
    if len(x.shape) < 4:
        AssertionError(f"Expect input shape len=4, received len={len(x.shape)}")
    z_i = torch.randn(1, 64, 1, 1)
    z_j = torch.randn(1, 64, 1, 1)
    P_i = torch.sqrt(torch.abs(gen(x, z_i)))
    P_j = torch.sqrt(torch.abs(gen(x, z_j)))
    
    L_freq = v * torch.linalg.norm(z_i - z_j, ord=2, dim=1) / torch.linalg.norm((torch.mean(P_i, dim=-2) - torch.mean(P_j, dim=-2)), ord=p_freq, dim=-1)
    L_rythm = v * torch.linalg.norm(z_i - z_j, ord=2, dim=1) / torch.linalg.norm((torch.mean(P_i, dim=-1) - torch.mean(P_j, dim=-1)), ord=p_rythm, dim=-1)

    return L_freq + L_rythm

# Initialize the weights using He initialization
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        fan_in = module.weight.size(1) * module.weight.size(2) * module.weight.size(3)
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

def model_size_log(logger, model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = n_params * 4 / 2 ** 20
    logger.info(f"{model.__class__.__name__}: parameters: {n_params}, size: {mb} MB")


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """

    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1 / self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)

def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])

def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")