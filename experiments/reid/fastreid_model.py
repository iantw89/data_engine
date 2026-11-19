"""Optional second candidate: fast-reid SBS R50-NL (MSMT17) loaded WITHOUT installing fast-reid: the repo is read
from disk (read-only) and unavailable third-party imports (termcolor, yacs...) are stubbed just enough to import the
backbone definition. Eval output = BNNeck feature (NECK_FEAT=after), input 384x128 RGB, 0-255 mean/std in the model."""
import sys, types
FASTREID = r"C:\Users\iantw\git\fast-reid"
WEIGHTS = FASTREID + r"\pretrained_download\msmt_sbs_R50.pth"


def _stub(name, **attrs):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m


class _Any:
    """Permissive stand-in for unavailable config libraries (never used for real work here)."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __setattr__(self, n, v): object.__setattr__(self, n, v)


def _stub_package(name):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    m = types.ModuleType(name)
    m.__path__ = []
    def _get(n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Any
    m.__getattr__ = _get
    sys.modules[name] = m
    return m


def build(device="cuda"):
    import torch
    from torch import nn
    sys.dont_write_bytecode = True
    import torchvision  # noqa: F401  (import before stubs)
    _stub("termcolor", colored=lambda s, *a, **k: s)
    for pkg in ("yacs", "yacs.config", "fvcore", "fvcore.common", "fvcore.common.checkpoint", "fvcore.common.file_io",
                "fvcore.nn", "tabulate", "faiss", "iopath", "iopath.common", "iopath.common.file_io", "cloudpickle"):
        _stub_package(pkg)
    if FASTREID not in sys.path:
        sys.path.insert(0, FASTREID)
    from fastreid.modeling.backbones.resnet import ResNet, Bottleneck
    from fastreid.layers import pooling

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet(1, "BN", False, False, True, Bottleneck, [3, 4, 6, 3], [0, 2, 3, 0])
            self.heads = nn.Module()
            self.heads.pool_layer = pooling.GeneralizedMeanPoolingP()
            self.heads.bnneck = nn.BatchNorm2d(2048)
            self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), False)

        def forward(self, x):  # x: (B,3,384,128) RGB 0..255
            x = (x - self.pixel_mean) / self.pixel_std
            f = self.backbone(x)
            f = self.heads.pool_layer(f)
            return self.heads.bnneck(f)[..., 0, 0]

    net = Net()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model"]
    state = {k: v for k, v in state.items() if not k.startswith("heads.classifier") and k not in ("pixel_mean", "pixel_std")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not [k for k in missing if k not in ("pixel_mean", "pixel_std")] and not unexpected, (missing, unexpected)
    return net.to(device).eval()
