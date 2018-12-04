from model.text.TextHandler import TextHandler

class ContainsText(TextHandler):
    '''
    A text handler for evaluating whether a user's query matches the text
    information associated with an audio file
    '''


    def set_query_text(self, query_text):
        '''
        Sets the value of the user's text-based query.

        Arguments:
            query_text: A string. The text associated with the user's query.
        '''
        self.query_text = query_text.lower()

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
        return self.query_text in text_features[0].lower()
