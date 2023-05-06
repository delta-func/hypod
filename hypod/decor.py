import importlib
import warnings
from dataclasses import MISSING, Field, is_dataclass, replace
from pathlib import Path
from typing import List, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml
from typeguard import TypeCheckError, check_type

from .parser import parse_argv

__all__ = ["MISSING", "hypod", "is_hypod", "is_union_of_hypod"]


def is_hypod(datacls):
    return hasattr(datacls, "_HYPOT") and is_dataclass(datacls)


def is_union_of_hypod(datacls):
    origin = get_origin(datacls)
    args = get_args(datacls)
    return origin is Union and any(is_hypod(arg) for arg in args)


def is_union_of(basetype, datacls):
    # Checks if `datacls` is Union[`basetype`, othertype1, othertype2, etc.]
    origin = get_origin(datacls)
    args = get_args(datacls)
    return origin is Union and basetype in args


class TypeCheckedDescriptor:
    def __init__(
        self, name, default, datacls, objcls, allow_parsing=True, update_if_hypod=True
    ):
        self._name = "_" + name
        self._default = default
        self.allow_parsing = allow_parsing
        self.update_if_hypod = update_if_hypod
        if allow_parsing and is_union_of(str, datacls):
            warnings.warn(
                f"The field '{name}' with type '{datacls}' accepts 'str' type, "
                "which arises ambiguity in parsing stringified objects. "
                "Here, the value will be force-treated as plain 'str'. "
                "Consider setting 'allow_parsing=False if you want different.'"
            )
        if is_union_of_hypod(datacls):
            # Check if Union of hypod have duplicate subcls names
            _keys = set()
            for _dc in get_args(datacls):
                if is_hypod(_dc):
                    _curr_n_keys = len(_keys)
                    _new_keys = set(_dc._subclasses)
                    _intersects = _keys.intersection(_new_keys)
                    if not len(_intersects) == 0:
                        raise ValueError(
                            f"The type {datacls} has duplicate tags: {_intersects}"
                        )
                    _keys.update(_new_keys)
            # Warn if datacls is Union[dict, Hypod]
            if dict in get_args(datacls):
                warnings.warn(
                    "Hypod class can be constructed from dict, "
                    "and thus making a Union with dict is not a good idea. "
                    "Here, if a dict value is given without '_tag' key,"
                    "it will be considered as plain 'dict'."
                )
        self._datacls = datacls
        self._objcls = objcls

    @property
    def name(self):
        return self._name[1:]

    @property
    def default(self):
        if isinstance(self._default, Field):
            if self._default.default is not MISSING:
                default_val = self._default.default
            if self._default.default_factory is not MISSING:
                assert (
                    self._default.default is MISSING
                ), "Only one of 'default' or 'default_factory' should be defined."
                default_val = self._default.default_factory()
            else:
                default_val = MISSING
        else:
            default_val = self._default
        return default_val

    def _check_if_MISSING(self, val):
        if val is MISSING:
            raise ValueError(
                "Default value is not defined for the field "
                f"'{self.name}' of '{self._objcls}'. Please provide a value."
            )

    def _check_type(self, val):
        try:
            check_type(val, self._datacls)
        except TypeCheckError:
            raise TypeCheckError(
                f"The value '{val}' with type '{type(val)}' given for "
                f"the field '{self.name}' of '{self._objcls}' is "
                f"not compatible with the annotated type '{self._datacls}'."
            )

    def __set__(self, obj, val):
        self._check_if_MISSING(val)
        # Parse stringified objects
        if isinstance(val, str):
            datacls_is_str_like = self._datacls is str or is_union_of(
                str, self._datacls
            )
            if self.allow_parsing and not datacls_is_str_like:
                if is_hypod(self._datacls) or is_union_of_hypod(self._datacls):
                    val = dict(_tag=val)
                else:
                    val = eval(val)  # val expected to be repr of the actual value

        # Handle Hypod objects
        if isinstance(val, dict):
            if is_hypod(self._datacls):
                if "_tag" in val:  # class hint is given by _tag
                    cls = self._datacls._subclasses[val["_tag"]]
                    val.pop("_tag")
                else:  # no class hint; use the annotated class
                    cls = self._datacls
                if self.update_if_hypod and self.default is not MISSING:
                    val = replace(self.default, **val)
                else:
                    val = cls(**val)
            elif is_union_of_hypod(self._datacls):
                if "_tag" in val:
                    # merge _subclasses dicts
                    _all_subclasses = {
                        k: v
                        for _datacls in get_args(self._datacls)
                        if is_hypod(_datacls)
                        for k, v in _datacls._subclasses.items()
                    }
                    cls = _all_subclasses[val["_tag"]]
                    val.pop("_tag")
                elif dict in get_args(self._datacls):
                    cls = dict
                else:
                    raise ValueError(
                        "A dict is given to construct a Hypod without a _tag key. "
                        "Since the attribute type annotation is Union, "
                        "its class cannot be determined."
                    )
                if self.update_if_hypod and self.default is not MISSING:
                    val = replace(self.default, **val)
                else:
                    val = cls(**val)

        # Check type and set value
        self._check_type(val)
        setattr(obj, self._name, val)

    def __get__(self, obj, objtype):
        if obj is None:
            return self._default
        else:
            return getattr(obj, self._name, self._default)


def _get_target_cls(cls):
    qualname_split = cls.__qualname__.rsplit(".", 1)
    if len(qualname_split) == 1:  # no outer class
        parent_cls = cls.__bases__[0]
        if parent_cls is not object:  # has a parent
            if not is_hypod(parent_cls):
                raise ValueError(
                    f"Hypod '{cls}' is inheriting a non-Hypod class '{parent_cls}'"
                )
            _target = _get_target_cls(parent_cls)
        else:
            _target = None  #
    else:
        module = importlib.import_module(cls.__module__)
        _target = getattr(module, qualname_split[0])
    return _target


T = TypeVar("T")


# def hypod(cls: Type[T]) -> Type[T]:
def hypod(cls) -> "cls":  # Pycharm works better with this annotation
    cls._HYPOT = "Hi I am a hypod"
    cls._subclasses = {}

    # Assign __init_subclass__ method to register child class
    def __init_subclass__(subcls, tag=None):
        try:
            super(cls, subcls).__init_subclass__(tag=tag)
        except TypeError:
            pass  # meaning: super is `object`, and we should pass.
        if tag is None:
            tag = subcls.__qualname__
        if tag in cls._subclasses:
            raise ValueError(f"The subclass tagged '{tag}' already exists for {cls}")
        cls._subclasses[tag] = subcls

    cls.__init_subclass__ = classmethod(__init_subclass__)

    # Define make() method which finds the output target class and make it
    def make(self):
        target_cls = _get_target_cls(cls)
        if target_cls is None:
            raise ValueError(
                "Unknown target. "
                "The outer class of the current or the parent hypod is not defined."
            )
        else:
            return target_cls(self)

    cls.make = make

    # Constructors
    def from_dict(_cls, arg_dict: dict) -> cls:
        return _cls(**arg_dict)

    cls.from_dict = classmethod(from_dict)

    def from_argv(_cls, argv: List[str]) -> cls:
        parsed_dict = parse_argv(argv)
        return _cls.from_dict(parsed_dict)

    cls.from_argv = classmethod(from_argv)

    def from_yaml(_cls, yaml_path: Union[str, Path]) -> cls:
        with open(yaml_path) as f:
            parsed_dict = yaml.load(f, Loader=yaml.FullLoader)
        return _cls.from_dict(parsed_dict)

    cls.from_yaml = classmethod(from_yaml)

    # Assign descriptors to handle:
    # 1) nested dataclasses
    # 2) parse stringified python objects with type-checking
    for var_name, var_type in get_type_hints(cls).items():
        var_val = getattr(cls, var_name, MISSING)
        setattr(
            cls,
            var_name,
            TypeCheckedDescriptor(
                name=var_name, default=var_val, datacls=var_type, objcls=cls
            ),
        )

    return cls
