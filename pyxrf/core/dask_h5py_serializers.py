# The original code for serializers/deserializers can be found in
# 'distributed/protocols/h5py.py'

import distributed.protocol.h5py  # noqa: F401
from distributed.protocol.serialize import dask_deserialize, dask_serialize

deserialized_files = set()


def serialize_h5py_file(f):
    if f and (f.mode != "r"):
        raise ValueError("Can only serialize read-only h5py files")
    filename = f.filename if f else None
    return {"filename": filename}, []


def serialize_h5py_dataset(x):
    header, _ = serialize_h5py_file(x.file if x else None)
    header["name"] = x.name if x else None
    return header, []


def deserialize_h5py_file(header, frames):
    import h5py

    filename = header["filename"]
    if filename:
        file = h5py.File(filename, mode="r")
        deserialized_files.add(file)
    else:
        file = None
    return file


def deserialize_h5py_dataset(header, frames):
    file = deserialize_h5py_file(header, frames)
    name = header["name"]
    dset = file[name] if (file and name) else None
    return dset


def dask_set_custom_serializers():
    import h5py

    dask_serialize.register((h5py.Group, h5py.Dataset), serialize_h5py_dataset)
    dask_serialize.register(h5py.File, serialize_h5py_file)
    dask_deserialize.register((h5py.Group, h5py.Dataset), deserialize_h5py_dataset)
    dask_deserialize.register(h5py.File, deserialize_h5py_file)


def dask_close_all_files():
    while deserialized_files:
        file = deserialized_files.pop()
        if file:
            file.close()
