from dataclasses import dataclass

from hypod import hypod


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


if __name__ == "__main__":
    # Hypod can be created in the same manner as dataclass.
    net_hp1 = NetworkHP(num_layers=3, layer_hp=LayerHP(64, 32))
    print(net_hp1)
    # Hypod can be also created from parsing stringified objects.
    net_hp2 = NetworkHP(num_layers="3", layer_hp=LayerHP("64", 32))
    print(net_hp2)
    # Hypod class itself can be also created from a dictionary.
    net_hp3 = NetworkHP(num_layers=3, layer_hp=dict(in_features=64, out_features=32))
    print(net_hp3)
