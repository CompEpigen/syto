"""App package marker.

Exists so that wizard submodules can be imported as ``App.wizard.*`` from the
repository root (used by the test suite). At runtime ``main.py`` is launched as
``python App/main.py``, which places ``App/`` on ``sys.path`` directly.
"""
