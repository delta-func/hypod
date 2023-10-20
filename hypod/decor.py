import importlib
import warnings
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import List, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml
from typeguard import TypeCheckError, check_type

from .parser import parse_argv

__all__ = ["MISSING", "hypod", "is_hypod", "is_union_of_hypod", "replace"]


def is_hypod(datacls):
    return hasattr(datacls, "_HYPOD") and is_dataclass(datacls)


def is_union_of_hypod(datacls):
    origin = get_origin(datacls)
    args = get_args(datacls)
    return origin is Union and any(is_hypod(arg) for arg in args)


def is_union_of(basetype, datacls):
    # Checks if `datacls` is Union[`basetype`, othertype1, othertype2, etc.]
    origin = get_origin(datacls)
    args = get_args(datacls)
    return origin is Union and basetype in args


def infer_new_hypod_cls(field_name, annotated_cls, curr_val, new_val_dict: dict):
    if is_hypod(annotated_cls):
        if "_tag" in new_val_dict:  # class hint is given by _tag
            cls = annotated_cls._subclasses[new_val_dict.pop("_tag")]
        elif curr_val is not MISSING:  # use the class of default
            cls = type(curr_val)
        else:
            cls = annotated_cls

    elif is_union_of_hypod(annotated_cls):
        if "_tag" in new_val_dict:
            # merge _subclasses dicts
            _all_subclasses = {
                k: v
                for _datacls in get_args(annotated_cls)
                if is_hypod(_datacls)
                for k, v in _datacls._subclasses.items()
            }
            cls = _all_subclasses[new_val_dict.pop("_tag")]
        elif curr_val is not MISSING:  # use the class of default
            cls = type(curr_val)
        else:
            raise ValueError(
                f"A dict is given to construct Hypod field '{field_name}' "
                "without a _tag key, while no default value has been set. "
                "Since the attribute type annotation is Union "
                f"'{annotated_cls}', "
                "its class cannot be determined."
            )

    else:
        raise ValueError(f"{field_name} is not a Hypod field")

    return cls


def infer_new_hypod_val(
    field_name, annotated_cls, curr_val, new_val: Union[dict, str], update_if_hypod=True
):
    if isinstance(new_val, str):
        new_val = dict(_tag=new_val)
    new_cls = infer_new_hypod_cls(field_name, annotated_cls, curr_val, new_val)
    if update_if_hypod and curr_val is not MISSING and new_cls is type(curr_val):
        new_val = replace(curr_val, update_if_hypod=update_if_hypod, **new_val)
    else:
        new_val = new_cls(**new_val)
    return new_val


def replace(obj, /, update_if_hypod=True, **changes):
    if not is_hypod(obj):
        raise TypeError("hypod's replace() should be called on hypod instances")

    for f in fields(obj):
        if not f.init:
            # Error if this field is specified in changes.
            if f.name in changes:
                raise ValueError(
                    f"field {f.name} is declared with "
                    "init=False, it cannot be specified with "
                    "hypod_replace()"
                )
            continue

        if f.name in changes:
            new_val = changes[f.name]
            if (is_hypod(f.type) or is_union_of_hypod(f.type)) and isinstance(
                new_val, (dict, str)
            ):  # implements unique updating behavior of hypod
                curr_val = getattr(obj, f.name)
                new_val = infer_new_hypod_val(
                    f.name, f.type, curr_val, new_val, update_if_hypod=update_if_hypod
                )
                changes[f.name] = new_val
        else:
            changes[f.name] = getattr(obj, f.name)

    return obj.__class__(**changes)


class TypeCheckedDescriptor:
    def __init__(
        self,
        name,
        default,
        datacls,
        objcls,
        allow_parsing=True,
        update_if_hypod=True,
        strict_type_checking=False,
    ):
        self._name = "_" + name
        self._default = default
        self.allow_parsing = allow_parsing
        self.update_if_hypod = update_if_hypod
        self.strict_type_checking = strict_type_checking
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
                raise ValueError(
                    "Hypod class can be constructed from dict, "
                    "and thus making a Union with dict is prohibited. "
                )
        self._datacls = datacls
        self._objcls = objcls

    @property
    def name(self):
        return self._name[1:]

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
            msg = (
                f"The value '{val}' with type '{type(val)}' given for "
                f"the field '{self.name}' of '{self._objcls}' is "
                f"not compatible with the annotated type '{self._datacls}'."
            )
            if self.strict_type_checking:
                raise TypeCheckError(msg)
            else:
                warnings.warn(msg)

    def __set__(self, obj, new_val):
        self._check_if_MISSING(new_val)
        # Parse stringified objects
        if (
            isinstance(new_val, str)
            and self.allow_parsing
            and not (self._datacls is str or is_union_of(str, self._datacls))
            and not (is_hypod(self._datacls) or is_union_of_hypod(self._datacls))
        ):
            new_val = eval(new_val)  # val expected to be repr of the actual value

        if (is_hypod(self._datacls) or is_union_of_hypod(self._datacls)) and isinstance(
            new_val, (dict, str)
        ):  # implements unique updating behavior of hypod
            curr_val = getattr(obj, self._name, self._default)
            new_val = infer_new_hypod_val(
                self.name, self._datacls, curr_val, new_val, self.update_if_hypod
            )

        # Check type and set value
        self._check_type(new_val)
        setattr(obj, self._name, new_val)

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
    cls._HYPOD = "Hi I am a hypod"
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
