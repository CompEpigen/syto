"""Helpers shared across task config assemblers."""


def set_nested(config: dict, dotted_key: str, value) -> None:
    """Write ``value`` into ``config`` at the path described by ``dotted_key``.

    ``"a.b.c"`` becomes ``config["a"]["b"]["c"] = value``, creating intermediate
    dicts as needed.
    """
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
