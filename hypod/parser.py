from typing import List


def check_argv_format(argv: List[str]):
    for a in argv:
        if "=" not in a or not (0 < a.index("=") < (len(a) - 1)):
            raise ValueError(f"argv '{a}' should be of the form 'foo.bar=baz'")


def parse_argv(argv: List[str]) -> dict:
    def _make_dict(_key, _val, base_dict):
        _key_split = _key.split(".", 1)
        if len(_key_split) == 2:  # e.g., foo.bar.baz=3
            _key, _key_rest = _key_split  # e.g., key=foo, _a_rest="bar.baz=3"
            if _key in base_dict:
                if not isinstance(base_dict[_key], dict):
                    base_dict[_key] = dict(_tag=base_dict[_key])
            else:
                base_dict[_key] = dict()
            _make_dict(_key_rest, _val, base_dict[_key])
        else:  # e.g., baz=3
            if _key in base_dict and isinstance(base_dict[_key], dict):
                base_dict[_key].update(_tag=_val)
            else:
                base_dict[_key] = _val

    check_argv_format(argv)
    argv_dict = dict()
    for a in argv:
        key, val = a.split("=")
        _make_dict(key, val, argv_dict)

    return argv_dict
