from dataclasses import dataclass

from hypod import hypod


@dataclass
@hypod
class Data:
    path: str
    batch_size: int = 4


@dataclass
@hypod
class FFHQData(Data, tag="ffhq"):
    path: str = "/data/public/rw/datasets/face/FFHQ"
    meta: str = "Flicker-Faces HQ Data, containing 70k images"


@dataclass
@hypod
class FFHQDataLargeBatch(FFHQData, tag="ffhq_lg"):
    batch_size: int = 16


@dataclass
@hypod
class LayerHP:
    in_features: int
    out_features: int
    init_scale: float = 0.9


@dataclass
@hypod
class NetworkHP:
    num_layers: int
    layer_hp: LayerHP


@dataclass
@hypod
class Model:
    data: Data
    net: NetworkHP


if __name__ == "__main__":
    model_with_ffhq = Model(
        data="ffhq", net=NetworkHP(num_layers=3, layer_hp=LayerHP(64, 32)),
    )
    print(model_with_ffhq)

    model_with_ffhq_lg = Model(
        data=dict(_tag="ffhq_lg"),  # Also can be created from dict with "_tag" key.
        net=NetworkHP(num_layers=3, layer_hp=LayerHP(64, 32)),
    )
    print(model_with_ffhq_lg)

    model_with_cifar10 = Model(
        data=dict(path="/path/to/cifar10"),
        net=NetworkHP(num_layers=3, layer_hp=LayerHP(64, 32)),
    )
    print(model_with_cifar10)
