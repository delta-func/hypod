from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypod import hypod, hypod_main, replace


@dataclass
@hypod
class Data:
    path: str
    batch_size: int = 4


@dataclass
@hypod
class FFHQData(Data, tag="ffhq"):
    path: str = "/path/to/FFHQ"


@dataclass
@hypod
class FFHQDataLargeBatch(FFHQData, tag="ffhq_lg"):
    batch_size: int = 16


class Layer(nn.Module):
    @dataclass
    @hypod
    class HP:
        in_features: Optional[int] = None
        out_features: Optional[int] = None
        activation: Union[bool, Callable] = True

        def __post_init__(self):
            if self.activation is True:
                self.activation = torch.relu
            elif self.activation is False:
                self.activation = lambda x: x

    hp: HP

    def __init__(self, hp: HP):
        super().__init__()
        self.hp = hp
        self.linear_layer = torch.nn.Linear(hp.in_features, hp.out_features)

    def forward(self, x):
        x = self.linear_layer(x)
        return self.hp.activation(x)


class Network(nn.Module):
    @dataclass
    @hypod
    class HP:
        in_dim: int
        out_dim: int
        num_layers: int
        mid_features: int = 128
        in_layer_hp: Layer.HP = field(default_factory=Layer.HP)
        mid_layer_hp: Layer.HP = field(default_factory=Layer.HP)
        out_layer_hp: Layer.HP = field(default_factory=Layer.HP)

        def __post_init__(self):
            # Whichever values have been set for the `in_features` and `out_features`
            # of the `in_layer_hp`, `mid_layer_hp`, and `out_layer_hp`, they will be
            # overwritten here to match with the "Network"-level hyperparameters:
            # `in_dim` and `out_dim`.
            self.in_layer_hp = replace(
                self.in_layer_hp,
                in_features=self.in_dim,
                out_features=self.mid_features,
            )
            self.mid_layer_hp = replace(
                self.mid_layer_hp,
                in_features=self.mid_features,
                out_features=self.mid_features,
            )
            self.out_layer_hp = replace(
                self.out_layer_hp,
                in_features=self.mid_features,
                out_features=self.out_dim,
            )

    hp: HP

    def __init__(self, hp: HP):
        self.hp = hp
        super().__init__()
        in_layer = Layer(hp.in_layer_hp)
        mid_layers = [Layer(hp.mid_layer_hp) for _ in range(hp.num_layers - 2)]
        out_layer = Layer(hp.out_layer_hp)
        self.layers = nn.ModuleList([in_layer] + mid_layers + [out_layer])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
@hypod
class ModelHP:
    data: Data
    net: Network.HP = field(
        default_factory=lambda: Network.HP(
            in_dim=32,
            out_dim=48,
            num_layers=4,
            in_layer_hp=Layer.HP(activation=F.sigmoid),
            out_layer_hp=Layer.HP(
                in_features=1234, activation=False
            ),  # in_features=1234 will be overwritten by __post_init__
        )
    )


@hypod_main()
def mymain(model_hp: ModelHP):
    print(model_hp)

    net1 = Network(model_hp.net)
    print(net1)

    # if Hypod is defined as an inner class, `make()` will automatically construct
    # the outer class object, feeding itself as the argument to the outer `__init__()`.
    net2 = model_hp.net.make()
    print(net2)


if __name__ == "__main__":
    mymain()
