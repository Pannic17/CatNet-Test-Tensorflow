import tarfile
import os

def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs)

if __name__ == "__main__":
    untar("annotations.tar.gz", ".")