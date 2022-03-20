# pylint: disable=W0221, E1102, W0703

from abc import ABC
import numpy as np
import torch

from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
import components
from utils.eval import model_loaders


class ESTIMATOR(
    BaseEstimator, ClassGradientsMixin, ClassifierMixin, LossGradientsMixin, ABC
):
    nb_classes = 2
    input_shape = ()

    def __init__(self, device, loader, config, loss=None):
        BaseEstimator.__init__(self, model=None, clip_values=(-1, 1))
        ClassGradientsMixin.__init__(self)
        ClassifierMixin.__init__(self)
        LossGradientsMixin.__init__(self)

        system, eer, logits, args = getattr(model_loaders, loader)(
            config, device, loss=loss
        )
        self.attack = getattr(components, system)(**args)
        self.device = device
        self.eer = eer
        self.logits = logits

    def fit(self, x, y, **kwargs) -> None:
        return

    def set_input_shape(self, value):
        setattr(self, "input_shape", value)

    def set_ref(self, x):
        try:
            self.attack.set_ref(x, self.device)

        except Exception:
            pass

    def class_gradient(self, x, label):
        if isinstance(x, np.ndarray):
            var = torch.tensor(
                x, device=self.device, dtype=torch.float, requires_grad=True
            )
        else:
            var = x

        score = self.attack.get_score(var)[:, label[0]]
        score.backward()
        grad = var.grad

        if isinstance(x, np.ndarray):
            grad = grad.detach().cpu().numpy()
        return grad

    def _loss(self, var, y):
        y = torch.tensor(np.argmax(y, axis=1), device=self.device).repeat(var.shape[0])
        loss = self.attack.attack_pipeline(var, y)
        return loss

    def compute_loss(self, x, y, reduction="mean"):
        # pylint: disable=W0613
        if isinstance(x, np.ndarray):
            var = torch.tensor(x, device=self.device, dtype=torch.float)
        else:
            var = x

        loss = self._loss(var, y)
        if isinstance(x, np.ndarray):
            loss = loss.detach().cpu().numpy()

        return loss

    def loss_gradient(self, x, y):
        if isinstance(x, np.ndarray):
            var = torch.tensor(
                x, device=self.device, dtype=torch.float, requires_grad=True
            )
        else:
            var = x

        loss = self._loss(var, y)
        loss.backward(retain_graph=True)
        grad = var.grad

        if isinstance(x, np.ndarray):
            grad = grad.detach().cpu().numpy()
        return grad

    def predict(self, x, logits=None, batch_size=1):
        # pylint: disable=W0613
        if isinstance(x, np.ndarray):
            var = torch.tensor(x, device=self.device, dtype=torch.float)
        else:
            var = x

        pred = self.attack.get_score(var, self.logits)[:, 1].squeeze()
        if x.shape[0] == 1:
            pred = pred.unsqueeze(0)

        # Find a better aggregation method for ADVJOINT (fixme)
        if len(pred.shape) > 1 and pred.shape[-1] > 1:
            ret = []
            for p in pred:
                if (p > self.eer).all():
                    ret.append([0.0, 1.0])
                else:
                    ret.append([1.0, 0.0])
            ret = torch.tensor(ret, device=self.device)
        if isinstance(x, np.ndarray):
            pred = pred.detach().cpu().numpy()
        return pred

    def result(self, x, label):
        if label.ndim > 1:
            label = np.argmax(label, axis=1)

        if isinstance(x, np.ndarray):
            out = x / np.max(np.abs(x), axis=-1)[..., np.newaxis]
            var = torch.tensor(out, device=self.device, dtype=torch.float)
        else:
            var = (x / torch.max(torch.abs(x), axis=-1).values.unsqueeze(-1)).float()

        score = self.attack.get_score(var, ret_logits=self.logits)[:, 1].squeeze(-1)

        if x.shape[0] == 1:
            score = score.unsqueeze(0)

        score = score.detach().cpu().numpy()
        result = [
            ((score[i] < self.eer).all() and label == 0)
            or ((score[i] >= self.eer).all() and label == 1)
            for i in range(len(score))
        ]

        ret = ["FAIL" if r == 0 else "SUCCESS" for r in result]
        return [
            (ret[i], str(score[i]) + "|" + str(self.eer)) for i in range(len(score))
        ]
