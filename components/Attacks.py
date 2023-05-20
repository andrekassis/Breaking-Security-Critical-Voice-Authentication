# pylint: disable=E1102

from abc import ABC
import torch
import torch.nn.functional as F


class BaseModel(ABC):
    def __init__(self, model, extractor, flip_label=False):
        self.model = model
        self.extract = extractor

        if flip_label:
            self._label = lambda x: 1 - x
            self._transform_output = lambda x: torch.flip(x, [-1])
        else:
            self._label = lambda x: x
            self._transform_output = lambda x: x

    def attack_pipeline(self, x, y):
        pass

    def get_score(self, x, ret_logits=True):
        pass

    def get_feats(self, x, **kwargs):
        return self.extract(x, **kwargs)


class ADVCM(BaseModel):
    def __init__(self, model, loss, extractor, flip_label=False):
        super().__init__(model, extractor, flip_label)
        self.loss = loss

    def attack_pipeline(self, x, y):
        y = self._label(y)
        feats = self.get_feats(x)
        y_pred = self.model(feats)
        loss = self.loss(y_pred, y).mean()
        return loss

    def get_score(self, x, ret_logits=True):
        feats = self.get_feats(x)
        output = self.model(feats, eval=True)
        output = self._transform_output(output)
        if ret_logits:
            return output
        output = F.softmax(output, dim=1)
        return output


class ADVSR(BaseModel):
    # pylint: disable=W0221
    def __init__(self, model, extractor, flip_label=False):
        super().__init__(model, extractor, flip_label)
        self.t_ivector = None

    def _extract_ivector(self, feats):
        ivector = self.model.Getivector(feats)
        trans_ivector = self.model.TransformIvectors(
            ivector, 1, simple_length_norm=False, normalize_length=True
        )
        return trans_ivector

    def set_ref(self, ref, device):
        y = torch.tensor(ref, device=device)
        t_feats = self.get_feats(y)
        self.t_ivector = torch.stack(
            [self._extract_ivector(t.unsqueeze(0)).squeeze(0) for t in t_feats]
        )

    def _get_batch(self, batch_sz):
        perm = torch.randperm(self.t_ivector.size(0))
        idx = perm[:batch_sz]
        return self.t_ivector  # [idx]

    def attack_pipeline(self, x, y, aggregate=True):
        batch = self._get_batch(50)  # batch_sz)
        feats = self.get_feats(x)
        testivector = self._extract_ivector(feats)
        loss = self.model.ComputePLDAScore(batch, testivector)
        factor = torch.ones(y.shape, device=loss.device).unsqueeze(-1)

        factor[y == 1] = -1
        ret = loss * factor

        if aggregate:
            ret = torch.mean(ret, axis=-1)
        ret = torch.mean(ret, axis=0)
        return ret

    def get_score(self, x, ret_logits=True):
        score = self.attack_pipeline(
            x, torch.tensor([1], device=x.device).repeat(x.shape[0]).unsqueeze(-1)
        )

        output = torch.stack((score, -score), axis=1)
        if ret_logits:
            return output
        output = F.softmax(output, dim=1)
        return output


class ADVJOINT:
    def __init__(self, asv_args, cm_args, lambda_asv, lambda_cm):
        self.asv = ADVSR(**asv_args)
        self.cm = ADVCM(**cm_args)
        self.lambda_asv = lambda_asv
        self.lambda_cm = lambda_cm

    def set_ref(self, ref, device):
        self.asv.set_ref(ref, device)

    def attack_pipeline(self, x, y):
        y_asv = self.asv.attack_pipeline(x, y)
        y_cm = self.cm.attack_pipeline(x, y)
        loss = self.lambda_asv * y_asv + self.lambda_cm * y_cm
        return loss

    def get_score(self, x, ret_logits=True):
        score_asv = self.asv.get_score(x, ret_logits[0])
        score_cm = self.cm.get_score(x, ret_logits[1])
        ret = torch.cat((score_asv, score_cm)).unsqueeze(-1)
        return ret
