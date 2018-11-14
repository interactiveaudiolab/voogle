import logging
from abc import ABC, abstractmethod

class TextHandler(ABC):
    '''
    Abstract base class for determining if the user's text query matches the
    text information associated with an audio file.
    '''
    def __init__(self):
        '''
        TextHandler constructor.
        '''
        self.logger = logging.getLogger('TextHandler')

        self.query_text = None

    def is_match(self, text_features):
        '''
        Returns true if text_features and query_text match. If query_text is an
        empty string, is_match will always return False.

        Arguments:
            text_features: A list of strings. The text features associated with
                an audio file.

        Returns:
            A boolean.
        '''
        if self.query_text == None:
            message = 'is_match called before call to set_query_text'
            self.logger.error(message)
            raise ValueError(message)
        return self._is_match(text_features)

    @abstractmethod
    def set_query_text(self, query_text):
        '''
        Sets the value of the user's text-based query.

        Arguments:
            query_text: A string. The text associated with the user's query.
        '''
        pass

    @abstractmethod
    def _is_match(self, text_features):
        '''
        Returns true if text_features and query_text match. If query_text is an
        empty string, is_match will always return True.

        Arguments:
            text_features: A list of strings. The text features associated with
                an audio file.

        Returns:
            A boolean.
        '''
        pass
