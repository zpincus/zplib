import pathlib
import pickle

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
    with path.open('wb') as f:
        pickle.dump(data_dict, f)

def load(path):
    """Load a dictionary from a pickle file into a Data object for
    attribute-style value lookup."""
    path = pathlib.Path(path)
    with path.open('rb') as f:
        return Data(**pickle.load(f))

def dump_csv(data, path):
    """Write a list of lists to a csv file."""
    path = pathlib.Path(path)
    with path.open('w') as f:
        f.write('\n'.join(','.join(row) for row in data))

def load_csv(path):
    """Load a csv file to a list of lists."""
    path = pathlib.Path(path)
    data = []
    with path.open('r') as f:
        for line in f:
            data.append(line.split(','))
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
