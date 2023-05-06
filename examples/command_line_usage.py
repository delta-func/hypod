from dataclasses import dataclass

from hypod import hypod, hypod_main


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


@hypod_main()  # parses sys.argv to construct the hypod in the first argument, `net_hp`
def main(net_hp: NetworkHP):
    print(net_hp)


if __name__ == "__main__":
    main()
