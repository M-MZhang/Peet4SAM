import os
import os.path as osp
import errno
import warnings
import json

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self,gt_path, high_path):
        assert check_isfile(gt_path)
        assert check_isfile(high_path)

        self._gtpath = gt_path
        self._highpath = high_path

    @property
    def gtpath(self):
        return self._gtpath

    @property
    def highpath(self):
        return self._highpath
        
    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname