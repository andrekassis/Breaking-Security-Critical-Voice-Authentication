import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlinearConcatGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(NonlinearConcatGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # F_squeeze
        self.fc1 = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(2 * int(channel / reduction), channel, bias=False), nn.Sigmoid()
        )

    def forward(self, x_prev, x):  # x: B*C*D*T
        b, c, _, _ = x.size()
        out1 = self.fc1(self.avg_pool(x_prev).view(b, c))
        out2 = self.fc2(self.avg_pool(x).view(b, c))
        out_cat = torch.cat([out1, out2], dim=1)
        y = self.fc_out(out_cat).view(b, c, 1, 1)

        return x_prev * y.expand_as(x_prev)


class LinearConcatGate(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearConcatGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(indim, outdim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_prev, x):
        x_cat = torch.cat([x_prev, x], dim=1)
        b, c_double, _, _ = x_cat.size()
        c = int(c_double / 2)
        y = self.avg_pool(x_cat).view(b, c_double)
        y = self.sigmoid(self.linear(y)).view(b, c, 1, 1)
        return x_prev * y.expand_as(x_prev)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: B*C*D*T
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEGatedNonlinearConcatBottle2neck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
        gate_reduction=4,
    ):

        super(SEGatedNonlinearConcatBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        if stype != "stage":
            gates = []
            for i in range(self.nums - 1):
                gates.append(NonlinearConcatGate(width, gate_reduction))
            self.gates = nn.ModuleList(gates)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = gate_sp + spx[i]
            sp = self.convs[i](sp)
            bn_sp = self.bns[i](sp)
            if self.stype != "stage" and i < self.nums - 1:
                gate_sp = self.gates[i](sp, spx[i + 1])
            sp = self.relu(bn_sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEGatedLinearConcatBottle2neck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
        gate_reduction=4,
    ):

        super(SEGatedLinearConcatBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        if stype != "stage":
            gates = []
            for i in range(self.nums - 1):
                gates.append(LinearConcatGate(2 * width, width))
            self.gates = nn.ModuleList(gates)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = gate_sp + spx[i]
            sp = self.convs[i](sp)
            bn_sp = self.bns[i](sp)
            if self.stype != "stage" and i < self.nums - 1:
                gate_sp = self.gates[i](sp, spx[i + 1])
            sp = self.relu(bn_sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(
        self,
        block,
        layers,
        baseWidth=26,
        scale=4,
        m=0.35,
        num_classes=1000,
        loss="softmax",
        **kwargs
    ):
        self.inplanes = 16
        super(Res2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, layers[0])  # 64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 512
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cls_layer = nn.Sequential(
            nn.Linear(128 * block.expansion, num_classes), nn.LogSoftmax(dim=-1)
        )
        self.loss_F = nn.NLLLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale)
            )

        return nn.Sequential(*layers)

    def _forward(self, x, eval=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cls_layer(x)

        return x

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    forward = _forward

    def save_state(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def optimizer(self, opt, **params):
        optimizer = getattr(torch.optim, opt)(
            filter(lambda p: p.requires_grad, self.parameters()), **params
        )
        return [optimizer]


class GatedRes2Net(nn.Module):
    def __init__(
        self,
        block,
        layers,
        baseWidth=26,
        scale=4,
        m=0.35,
        num_classes=1000,
        loss="softmax",
        gate_reduction=4,
        **kwargs
    ):
        self.inplanes = 16
        super(GatedRes2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.gate_reduction = gate_reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, layers[0])  # 64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 512
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cls_layer = nn.Sequential(
            nn.Linear(128 * block.expansion, num_classes), nn.LogSoftmax(dim=-1)
        )
        self.loss_F = nn.NLLLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
                gate_reduction=self.gate_reduction,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                    gate_reduction=self.gate_reduction,
                )
            )

        return nn.Sequential(*layers)

    def _forward(self, x, eval=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cls_layer(x)

        return x

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    forward = _forward

    def save_state(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def optimizer(self, opt, **params):
        optimizer = getattr(torch.optim, opt)(
            filter(lambda p: p.requires_grad, self.parameters()), **params
        )
        return [optimizer]


class SEBottle2neck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
    ):

        super(SEBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_gated_linearconcat_res2net50_v1b(**kwargs):
    model = GatedRes2Net(
        SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs
    )
    return model


def se_gated_nonlinearconcat_res2net50_v1b(**kwargs):
    model = GatedRes2Net(
        SEGatedNonlinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs
    )
    return model


def se_res2net50_v1b(**kwargs):
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model


def Detector(MODEL_SELECT, GATE_REDUCTION=4):
    if MODEL_SELECT == 25:
        model = se_gated_linearconcat_res2net50_v1b(
            num_classes=2, KaimingInit=True, gate_reduction=GATE_REDUCTION
        )
    elif MODEL_SELECT == 31:
        model = se_gated_nonlinearconcat_res2net50_v1b(
            num_classes=2, KaimingInit=True, gate_reduction=GATE_REDUCTION
        )
    elif MODEL_SELECT == 5:
        model = se_res2net50_v1b(num_classes=2, KaimingInit=True)
    return model
