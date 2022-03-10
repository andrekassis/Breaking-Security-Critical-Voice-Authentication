# pylint: disable=C0103,W0613,R0201,R0913

import torch
import numpy as np


class CompScheduler:
    def __init__(self, optimizer, lr_decay, lr_stepLR_size):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=lr_stepLR_size, gamma=lr_decay
        )

    def update(self, lr, epoch_num, step):
        if step:
            self.lr_scheduler.step()

    def increase_delta(self):
        return


class AIRScheduler:
    def __init__(self, optimizer, lr_decay, interval):
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.interval = interval

    def update(self, lr, epoch_num, step):
        if step:
            lr = lr * (self.lr_decay ** (epoch_num // self.interval))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def increase_delta(self):
        return


class SSNetScheduler:
    def __init__(self, optimizer, gamma):
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )

    def update(self, lr, epoch_num, step):
        if step:
            self.lr_scheduler.step()

    def increase_delta(self):
        return


class DartsScheduler:
    def __init__(self, optimizer, num_epochs, lr_min):
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(num_epochs), eta_min=lr_min
        )

    def update(self, lr, epoch_num, step):
        if step:
            self.lr_scheduler.step()

    def increase_delta(self):
        return


class rawGATScheduler:
    def __init__(self, optimizer):
        return

    def update(self, lr, epoch_num, step):
        return

    def increase_delta(self):
        return


class Res2NetScheduler:
    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def update(self, lr, epoch_num, step):
        if step:
            return

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return

    def increase_delta(self):
        self.delta *= 2


class AASSISTScheduler:
    def __init__(self, optimizer, n_epochs, steps_per_epoch, base_lr, lr_min):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self.cosine_annealing(
                step, n_epochs * steps_per_epoch, 1, lr_min / base_lr
            ),
        )

    @staticmethod
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi)
        )

    def update(self, lr, epoch_num, step):
        if step:
            self.lr_scheduler.step()

    def increase_delta(self):
        return
