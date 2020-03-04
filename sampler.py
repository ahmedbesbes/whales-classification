import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):

    def __init__(self, data_source, classes, labels_to_samples, mapping_files_to_global_id, p=64, k=16):
        super().__init__(data_source)
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
                    path = f'../data/train/{l}/{s}'
                    index = self.mapping_files_to_global_id[path]
                    yield index

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples
