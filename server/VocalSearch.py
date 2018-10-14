import numpy as np


class VocalSearch(object):
    '''
    TODO
    '''

    def __init__(self, model, dataset):
        '''
        VocalSearch constructor

        Arguments:
            model: A QueryByVoiceModel. The model used to perform similarity
                calculations
            dataset: A python generator. The generator used to load audio
                representations for similarity ranking.
        '''
        self.model = model
        self.dataset = dataset

    def search(self, query, sampling_rate):
        '''
        TODO
        '''
        # Construct query representation
        query = self.model.construct_representation(
            [query], [sampling_rate], True)

        # run model inference and return the matching files in order of
        # similarity
        return self.inference(query[0], self.model, self.dataset)

    def inference(self, query):
        '''
        TODO
        '''
        # Retrieve the similarity measure between query and each dataset entry
        model_output = []
        filenames = []
        for batch, batch_filenames in self.dataset:
            model_output.append(self.model.predict(query, batch))
            filenames.concatenate(batch_filenames)
        model_output = np.array(model_output).flatten()

        # Determine ranking
        sorted_index = np.argsort(model_output.flatten())[::-1]

        # Sort the files by similarity rank
        return filenames[sorted_index].to_list()
