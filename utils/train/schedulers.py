# pylint: disable=C0103,W0613,R0201,R0913

import torch
import numpy as np
from . import schedulers


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


"""
class SENet_FFT_scheduler:
    def __init__(self, optimizer, n_warmup):
        self.optimizer = optimizer
        self.n_warmup = n_warmup
        self.steps = 0
    def update(self, lr, epoch_num, step):
        if step:
            self.steps += 1
            if self.steps <= self.n_warmup:
                1/np.sqrt(self.steps)
            lr = lr * (self.lr_decay ** (epoch_num // self.interval))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def increase_delta(self):
        return
"""


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


class GE2EScheduler:
    def __init__(self, optimizer, gamma=0.95):
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )

    def update(self, lr, epoch_num, step):
        if step:
            self.lr_scheduler.step()

    def increase_delta(self):
        return


"""
class GE2EScheduler:
    def __init__(self, optimizer):
        return

    def update(self, lr, epoch_num, step):
        return

    def increase_delta(self):
        return
"""


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


class WAV2VECScheduler:
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


class OptimizerWrapper:
    def __init__(self, optimizer, sched):
        # pylint: disable=W0621
        self.optimizer = optimizer
        self.lr_scheduler = [
            getattr(schedulers, sched["type"])(opt, **sched["params"])
            for opt in self.optimizer
        ]

    def update(self, lr, epoch_num, step):
        # pylint: disable=W0621
        for s in self.lr_scheduler:
            s.update(lr, epoch_num, step)

    def zero_grad(self):
        for o in self.optimizer:
            o.zero_grad()

    def step(self):
        for o in self.optimizer:
            o.step()

    def increase_delta(self):
        for s in self.lr_scheduler:
            s.increase_delta()

    def load(self, itrs, results, lr, num_batch, val_metric):
        for epoch_num in range(itrs):
            for t in range(num_batch):
                self.zero_grad()
                self.step()
                self.update(lr, epoch_num, False)
            self.update(lr, epoch_num, True)

            res = results[epoch_num]
            if epoch_num == 0:
                best_res = res
                best_epoch = epoch_num
                best_epoch_tmp = epoch_num

            is_best = (
                res < best_res if val_metric in ("eer", "loss") else res > best_res
            )
            if is_best:
                best_epoch = epoch_num
                best_epoch_tmp = epoch_num
                best_res = res

            if epoch_num - best_epoch_tmp > 2:
                self.increase_delta()
                best_epoch_tmp = epoch_num
