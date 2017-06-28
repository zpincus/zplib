import json
import numpy
import os
import pathlib
import pickle
import tempfile

def get_dir(path):
    """Create a directory at path if it does not already exist. Return a
    pathlib.Path object for that directory."""
    path = pathlib.Path(path)
    if path.exists():
        if not path.is_dir():
            raise RuntimeError('Path {} is not a directory.'.format(str(path)))
    else:
        path.mkdir(parents=True)
    return path

def dump(path, **data_dict):
    """Dump the keyword arguments into a dictionary in a pickle file."""
    path = pathlib.Path(path)
    try:
        with path.open('wb') as f:
            pickle.dump(data_dict, f)
    except:
        if path.exists():
            path.remove()

def load(path):
    """Load a dictionary from a pickle file into a Data object for
    attribute-style value lookup. The path to the original file
    from which the data was loaded is stored in the '_path' attribute."""
    path = pathlib.Path(path)
    with path.open('rb') as f:
        return Data(_path=path, **pickle.load(f))

def dump_csv(data, path):
    """Write a list of lists to a csv file."""
    path = pathlib.Path(path)
    try:
        with path.open('w') as f:
            f.write('\n'.join(','.join(row) for row in data))
    except:
        if path.exists():
            path.remove()

def load_csv(path):
    """Load a csv file to a list of lists."""
    path = pathlib.Path(path)
    data = []
    with path.open('r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data

class Data:
    def __init__(self, **kwargs):
        """Add all keyword arguments to self.__dict__, which is to say, to
        the namespace of the class. I.e.:

        d = Data(foo=5, bar=6)
        d.foo == 5 # True
        d.bar > d.foo # True
        d.baz # AttributeError
        """
        self.__dict__.update(kwargs)

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that is smart about converting iterators and numpy arrays to
    lists, and converting numpy scalars to python scalars.
    """
    def default(self, o):
        try:
            return super().default(o)
        except TypeError as x:
            if isinstance(o, numpy.generic):
                item = o.item()
                if isinstance(item, numpy.generic):
                    raise x
                else:
                    return item
            try:
                return list(o)
            except:
                raise x


_COMPACT_ENCODER = _NumpyEncoder(separators=(',', ':'))
_READABLE_ENCODER = _NumpyEncoder(indent=4, sort_keys=True)

def json_encode_compact_to_bytes(data):
    """Encode compact JSON for transfer over the network or similar."""
    return _COMPACT_ENCODER.encode(data).encode('utf8')

def json_encode_legible_to_str(data):
    """Encode nicely-formatted JSON to a string."""
    return _READABLE_ENCODER.ncode(data)

def json_encode_legible_to_file(data, f):
    """Encode nicely-formatted JSON to an open file handle."""
    for chunk in _READABLE_ENCODER.iterencode(data):
        f.write(chunk)

def json_encode_atomic_legible_to_file(data, filename, suffix=None):
    """Encode nicely-formatted JSON and if there was no error, atomically write.

    Care is taken to never overwrite an existing file except in an atomic manner
    after all other steps have occured. This prevents errors from causing a
    partial overwrite of an existing file: the result of this function is all or
    none.

    Parameters:
        data: python objects to be JSON encoded
        filename: string or pathlib.Path object for destination file.
        suffix: if provided, the temporary file will have the same name as
            the original file, plus this suffix and then some arbitrary characters
            to ensure uniqueness. If not provided, the suffix will be 'temp'.
            If a suffix is provided, in the event of a file-write error, the
            partially written temp file will be left for possible user-recovery.
            Otherwise, the file will be removed if an error occurs.
    """
    s = json_encode_legible_to_str(data)
    filename = pathlib.Path(filename)
    prefix = filename.name + '-{}.'.format('temp' if suffix is None else suffix)
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, dir=str(filename.parent))
    try:
        with os.fdopen(fd) as f:
            f.write(s)
        os.replace(str(tmp_path), str(filename))
    except:
        if not suffix:
            # if no suffix provided, assume that there's no interest in user-recovery
            # of half-written files.
            os.remove(tmp_path)
        raise