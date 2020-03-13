import itertools
import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, root, data_source, classes, labels_to_samples, mapping_files_to_global_id, p=64, k=16):
        super().__init__(data_source)
        self.root = root
        self.p = p
        self.k = k
        self.data_source = data_source
        self.classes = classes
        self.labels_to_samples = labels_to_samples
        self.mapping_files_to_global_id = mapping_files_to_global_id

    def __iter__(self):
        pk_count = len(self) // (self.p * self.k)
        for _ in range(pk_count):
            labels = np.random.choice(self.classes,
                                      self.p,
                                      replace=False)

            for l in labels:
                samples = self.labels_to_samples[l]
                replace = True if len(samples) < self.k else False
                for s in np.random.choice(samples, self.k, replace=replace):
                    path = f'{self.root}{l}/{s}'
                    index = self.mapping_files_to_global_id[path]
                    yield index

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples


# sampler 2

def grouper(iterable, n):
    it = itertools.cycle(iter(iterable))
    for _ in range((len(iterable) - 1) // n + 1):
        yield list(itertools.islice(it, n))

# full label coverage per 'epoch'


class PKSampler2(Sampler):

    def __init__(self, root, data_source, classes, labels_to_samples, mapping_files_to_global_id, p=64, k=16):
        super().__init__(data_source)
        self.root = root
        self.p = p
        self.k = k
        self.data_source = data_source
        self.classes = classes
        self.labels_to_samples = labels_to_samples
        self.mapping_files_to_global_id = mapping_files_to_global_id

    def __iter__(self):
        rand_labels = np.random.permutation(
            np.arange(len(self.labels_to_samples.keys())))
        for labels_indices in grouper(rand_labels, self.p):
            for li in labels_indices:
                label = list(self.labels_to_samples.keys())[li]
                samples = self.labels_to_samples[label]
                replace = True if len(samples) < self.k else False
                for s in np.random.choice(samples, self.k, replace=replace):
                    path = f'{self.root}{l}/{s}'
                    index = self.mapping_files_to_global_id[path]
                    yield index

    def __len__(self):
        num_labels = len(set(self.classes))
        samples = ((num_labels - 1) // self.p + 1) * self.p * self.k
        return samples
