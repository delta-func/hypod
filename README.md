# Hypod: A dataclass-based hyperparameter managing system

## Overview
Deep learning models are often composed of multiple networks, where each of the networks is composed of multiple layers, and the class of each layer or its hyperparameters differ by experiments. `Hypod` simplifies managing this complex hierarchy of hyperparameters utilizing the built-in `dataclass`.

The `dataclass` derives the following benefits.
* Defining a group of hyperaparameters is easy with **type annotation** support.
* Creating it is easy with the auto-defined `__init__`.
* Many IDE's (e.g., PyCharm, VSCode) support "jump to the definition".

However, difficulties for using it "as-is" in hyperparameter-managing are:
* Handling nested dataclasses is not so good.
* Parsing the strings or sys.argv to create a dataclass is not natively supported.
* Switching between multiple child dataclasses using a simple "tag" is cumbersome.

`Hypod` will handle all these difficulties for you with following advantages.
* Fast, lightweight implementation using the built-in `dataclass` and descriptors
* Minimal dependency
* Type-checking (based on annotation)
* Parsing a stringified object or YAML to create the corresponding nested dataclass
* Ability to auto-make the corresponding module (e.g., layer, network), when defined as an inner class of that module.

### Comparison with other packages
* [Hydra](https://github.com/facebookresearch/hydra): Hydra is a popular package with the same purpose. Its _structured config_ mode allows defining a config with `dataclass` as well. However, Hydra converts a config into `DictConfig` object (from another package `omegaconf`) even when it is originally defined with `dataclass`. Thus, all the operations (modifying, getting values, merging) are done with `DictConfig` and this brings the following drawbacks compared to Hypod, which uses `dataclass` object all the time.
  * `DictConfig` is not type annotated.
  * "Go to the definition" in IDE cannot be done with `DictConfig`.
  * Inheritance strcuture of configs cannot be checked.
  * Complex value interpolation is difficult or impossible to implement. In Hypod `dataclass`, it can be done using built-in `__post_init__()` function (see `examples/advanced_usage.py`).


### Etymology
Hypod stands for **"A Pot of Hyperparameters, or A Hyperparameters-Pot"**. You can put various Hyperparameters in a Pot and mix them with others as you wish.

## Quick start
### Install the package
`pip install hypod`

### Basic Usage

```python
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

```

### Command-line Usage
With the Hypod classes defined the same as above, define the main function as follows.

```python
# main.py
from hypod import hypod_main


@hypod_main()  # parses sys.argv to construct the hypod in the first argument, `net_hp`
def main(net_hp: NetworkHP):
  print(net_hp)


if __name__ == "__main__":
  main()
```
Then, in the command-line type as follows to obtain the same results as before.
`python main.py num_layers=3 layer_hp.in_features=64 layer_hp.out_features=32`


### Inheritance
Hypod can be subclassed with a **tag**.

```python
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
  path: str = "/path/to/FFHQ"
  meta: str = "Flicker-Faces HQ Data, containing 70k images"


@dataclass
@hypod
class FFHQDataLargeBatch(FFHQData, tag="ffhq_lg"):
  batch_size: int = 16


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
```

### Command-line Usage with Inheritance
With the Hypod classes defined the same as above, define the main function as follows.

```python
# main.py
from hypod import hypod_main


@hypod_main()  # parses sys.argv to construct the hypod in the first argument, `net_hp`
def main(model: Model):
  print(model)


if __name__ == "__main__":
  main()
```
Then, in the command-line type as follows to obtain the same results as before.
`python main.py model.data=ffhq model.net.num_layers=3 model.net.layer_hp.in_features=64 model.net.layer_hp.out_features=32`
I.e., tag can be fed in the CLI as the root argument.

### Advanced Usage (Value Interpolation)
* Please refer to `advanced_usage.py` in the `examples` directory.
* Please note that manually updating the fields of a Hypod object is prohibited. You should always use [`replace`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace) which will automatically call `__post_init__()` to correctly process the value interpolation.



### FAQ
* (Q) When creating Hypod from dict, how does it know which hypod to create? 
  * (A) Via type annotation.
* (Q) Is `typing.Union` supported? 
  * (A) Yes. But when it is a union of `str` or `dict`, there are ambiguities in parsing the stringified objects. A warning or an error will be raised in this case.
* (Q) Does tag-based creation still work when `typing.Union` is used?
  * (A) Yes. The tag search will be done over all the types in the union.
* (Q) Are `typing.List` or other generic types supported?
  * (A) Yes.
* (Q) How can I use value interpolation?
  * (A) `dataclass` provides `__post_init__()` function where you can define a .
* (Q) Why not using `typing.dataclass_transform()` in implementing Hypod?
  * (A) Because requirement of Python>=3.11 is too strict for now.