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


def placeholder(name: str) -> str:
    """Return the marker written into a template for a value the user must supply.

    Angle brackets are the single convention for "fill this in": template mode
    counts them, and the generated file's header tells the reader to grep for
    them. Keep them free of ``:`` so the YAML stays unquoted and readable.
    """
    return f"<{name}>"


def is_placeholder(value) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")
