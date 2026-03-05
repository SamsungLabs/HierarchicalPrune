# pylint: skip-file

try:
    from attrdict import AttrDict
except ImportError:
    import collections
    import collections.abc

    for type_name in collections.abc.__all__:  # noqa
        setattr(collections, type_name, getattr(collections.abc, type_name))  # noqa
    from attrdict import AttrDict
