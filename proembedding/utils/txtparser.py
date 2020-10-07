class ExcludeTxt:

    def __init__(self, path):
        self.path = path

    @property
    def indices_set(self):
        try:
            return self._indices_set
        except AttributeError:
            self._indices_set = self._get_indices()
            return self._indices_set

    def _get_indices(self):
        with open(self.path, "r") as f:
            lines = f.readlines()
        indices = set([])
        for line in lines:
            try:
                index = line.strip().split(":")[1]
                indices.add(index)
            except IndexError:
                continue
        return indices
