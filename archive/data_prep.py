import numpy as np
import torch.utils.data.dataset as dataset

class InferenceData(dataset.Dataset):
	def __init__(self, imitation, references):
		self.references = references
		self.imitation = imitation
		# self.pairs = [[imitation, r] for r in references]

	def __getitem__(self, index):
		return self.imitation, self.references[index]

	def __len__(self):
		return len(self.references)