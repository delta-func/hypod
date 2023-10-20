import functools
import logging
import pprint
import sys
from pathlib import Path
from typing import List, Union, get_type_hints

import yaml

from .decor import is_hypod
from .parser import parse_argv


def parse_opts_as_dict(opts: List[str]) -> dict:
    opt_dicts = dict()
    for opt in opts:
        assert opt.startswith("--")
        opt = opt[2:]
        if "=" not in opt or not (0 < opt.index("=") < (len(opt) - 1)):
            raise ValueError(f"System option '{opt}' should be of the form '--foo=bar'")
        key, val = opt.split("=")
        opt_dicts[key] = val
    return opt_dicts


def nested_dict_update(orig_nested_dict: dict, update_nested_dict: dict):
    def _dict_update(_orig, _update):
        for k, v in _update.items():
            if k in _orig and isinstance(_orig[k], dict):
                _dict_update(_orig[k], v)
            else:
                _orig.update({k: v})

    _dict_update(orig_nested_dict, update_nested_dict)


def hypod_main(
    yaml_pre: Union[str, Path, None] = None,
    yaml_post: Union[str, Path, None] = None,
    logger=None,
):
    def _main_decor(main_fn):
        @functools.wraps(main_fn)
        def _wrapped_main_fn():
            if logger is None:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    fmt="%(asctime)s [%(name)s/%(levelname)s] %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                )
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.INFO)

                hypod_logger = logging.getLogger("hypod")
                hypod_logger.addHandler(console_handler)
            else:
                hypod_logger = logger
            type_hints = get_type_hints(main_fn)
            type_hints.pop("return", None)

            if len(type_hints) == 1:
                var_name = list(type_hints)[0]
                var_type = type_hints[var_name]
            else:
                var_type = None

            if var_type is None or not is_hypod(var_type):
                raise TypeError(
                    f"{main_fn} decorated by 'hypod.main' should take "
                    "a single argument with Hypod-class annotated."
                )

            sys_argv = []  # e.g., foo.bar=baz
            sys_opts = []  # e.g., --foo=bar
            for a in sys.argv[1:]:
                if a.startswith("--"):  # system options start with a double-dash.
                    sys_opts.append(a)
                else:
                    sys_argv.append(a)
            sys_opts = parse_opts_as_dict(sys_opts)
            args_dict = dict()

            def _load_yaml_and_update_dict(
                given_yaml_path=None, default_yaml_path=None
            ):
                if default_yaml_path is not None:
                    with open(default_yaml_path) as f:
                        parsed_dict = yaml.load(f, Loader=yaml.FullLoader)
                        nested_dict_update(args_dict, parsed_dict)
                if given_yaml_path is not None:
                    with open(given_yaml_path) as f:
                        parsed_dict = yaml.load(f, Loader=yaml.FullLoader)
                        nested_dict_update(args_dict, parsed_dict)

            # Note The Priority (The former will be overwritten by the latter):
            # * Hypod (dataclass) definition in python
            # * yaml_pre from hypod_main arguments
            # * --yaml_pre from sys_opts
            # * sys_argv (e.g., foo.bar=baz)
            # * yaml_post from hypod_main arguments
            # * --yaml_post from sys_opts

            # Yaml-pre
            _load_yaml_and_update_dict(
                given_yaml_path=sys_opts.get("yaml_pre"), default_yaml_path=yaml_pre
            )
            # Main Hypod
            nested_dict_update(args_dict, parse_argv(sys_argv))
            # Yaml-post
            _load_yaml_and_update_dict(
                given_yaml_path=sys_opts.get("yaml_post"), default_yaml_path=yaml_post
            )

            hypod_instance = var_type.from_dict(args_dict)
            hypod_logger.info(pprint.pformat(hypod_instance))
            # pprint.pprint(hypod_instance)
            return main_fn(hypod_instance)

        return _wrapped_main_fn

    return _main_decor
