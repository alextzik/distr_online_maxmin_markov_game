import numpy as np


class Partition(object):
    """Class representation of partitions."""
    __slots__ = ['npoints', 'ncells', 'cells', 'extra']

    def __init__(self, npoints, ncells, cells):
        """Creates a partition over the set {0,...,npoints-1}.
        The partition does not have to be consistent at the stage yet.

        :param npoints: the size of the partitioned set
        :param ncells: the number of cells
        :param cells: the cells
        :return: Partition object with the provided attributes
        """
        self.npoints = npoints
        self.ncells = ncells
        assert ncells <= npoints
        self.cells = cells
        self.extra = {}

    def cell_sizes(self):
        """Returns the number of elements in each cell.

        :return: integer tuple representing the cell sizes
        """
        sizes = []
        for cell in self.cells:
            sizes.append(len(cell))
        return tuple(sizes)

    def cell_indices(self):
        """Returns the cell index for each element of the partitioned set.

        :return: integer array representing the cell index of each element
        """
        idx = np.empty(self.npoints, dtype=int)
        for i, cell in enumerate(self.cells):
            idx[cell] = i
        return idx

    def assert_consistency(self):
        """Raises an assertion error if the partition is not consistent."""
        assert self.ncells == len(self.cells)
        elems = []
        for k in range(self.ncells):
            cell_k = self.cells[k]
            assert 0 < len(cell_k)
            elems += list(cell_k)
        assert self.npoints == len(elems)
        assert list(range(self.npoints)) == sorted(elems)

    def __eq__(self, other):
        """Test whether two partition are equal (assumes assert_consistency() passes for both partitions)."""
        if not isinstance(other, Partition):
            return NotImplementedError('Cannot compare Partition to {}!'.format(type(other)))

        if self.npoints != other.npoints or self.ncells != other.ncells:
            return False
        for cell, other_cell in zip(sorted([tuple(c) for c in self.cells]),
                                    sorted([tuple(c) for c in other.cells])):
            if len(cell) != len(other_cell):
                return False
            if tuple(cell) != tuple(other_cell):
                return False
        return True


def max_affine_partition(data, maf):
    """Returns the induced partition by a max-affine function.

    :param data: data matrix (each row is a sample)
    :param maf: max-affine function as a matrix (each row is an affine map [offset, slope])
    :returns: Partition object representing the induced partition

    """
    nhyperplanes = maf.shape[0]
    idx = np.argmax(data.dot(maf.T), axis=1)
    cells = []
    for k in range(nhyperplanes):
        cells.append(np.where(idx == k)[0])
    cells = [c for c in cells if len(c) > 0]
    return Partition(npoints=data.shape[0], ncells=len(cells), cells=tuple(cells))
