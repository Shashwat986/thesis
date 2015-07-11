# Cooccurrence matrix construction tools
# for fitting the GloVe model.
import numpy as np
import scipy.sparse as sp
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

from .corpus_cython import construct_cooccurrence_matrix, add_sentences


class Corpus(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.

    A dictionry mapping words to ids can optionally
    be supplied. If left None, it will be constructed
    from the corpus.
    """
    
    def __init__(self, dictionary=None):

        self.dictionary = {}
        self.dictionary_supplied = False
        self.matrix = None
        self.unigram = None
        self.xvalues = None

        if dictionary is not None:
            self._check_dict(dictionary)
            self.dictionary = dictionary
            self.dictionary_supplied = True

    def _check_dict(self, dictionary):

        if (np.max(list(dictionary.values())) != (len(dictionary) - 1)):
            raise Exception('The largest id in the dictionary '
                            'should be equal to its length minus one.')

        if np.min(list(dictionary.values())) != 0:
            raise Exception('Dictionary ids should start at zero')

    def fit(self, corpus, window=10, max_map_size=1000, ignore_missing=False):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix. 

        Parameters:
        - iterable of lists of strings corpus
        - int window: the length of the (symmetric)
          context window used for cooccurrence.
        - int max_map_size: the maximum size of map-based row storage.
                            When exceeded a row will be converted to
                            more efficient array storage. Setting this
                            to a higher value will increase speed at
                            the expense of higher memory usage.
        - bool ignore_missing: whether to ignore words missing from
                               the dictionary (if it was supplied).
                               Context window distances will be preserved
                               even if out-of-vocabulary words are
                               ignored.
                               If False, a KeyError is raised.
        """
        
        self.matrix,self.unigram,self.xvalues = construct_cooccurrence_matrix(corpus,
                                       self.dictionary,
                                       int(self.dictionary_supplied),
                                       int(window),
                                       int(ignore_missing),
                                       max_map_size)
        print self.matrix.shape, "Shape"

    def add_sentences(self, sentences, length = None):
        """
        Adds sentences to the cooccurrence matrix based on the following algorithm:
        
        Parameters:
        - Iterable of list of sentences
        - length of list (if a generator is used)
        """

        
        if type(sentences) == type([]):
            length = len(sentences)
        elif length is None:
            raise TypeError("Need to specify length for generators")

        self.matrix,self.dictionary = add_sentences(
                        self.matrix, self.dictionary, self.unigram, self.xvalues,
                        sentences, length)

        """        
        csc_matrix = self.matrix.tocsc()
        d_len = len(self.dictionary)
        shape = d_len + length
        new_row = []
        new_col = []
        new_data = []

        # We need to calculate Matrix[w,sent] For every word w in dictionary
        # For every sentence in sentences
        for i,sentence in enumerate(sentences):
            print i
            if i % int(length / 20) == 0: print "{}/{}".format(i,length)
            # For every word in dictionary
            for d_word in self.dictionary:
                d_id = self.dictionary[d_word]

                # Algorithm to calculate Matrix[d_word, sentence]
                cooc = 0.0
                num_s_words = 0
                for s_word in sentence:
                    sw_id = self.dictionary.get(s_word,-1)
    
                    if sw_id == -1 or d_id == sw_id:
                        continue
                    else:
                        num_s_words += 1
                        if d_id < sw_id:
                            cooc += csc_matrix[d_id, sw_id]
                        else:
                            cooc += csc_matrix[sw_id, d_id]
                    
                if num_s_words > 0:
                    cooc /= num_s_words
                # Algorithm ends
    
                if cooc != 0.0:
                    new_row.append(self.dictionary[d_word])
                    new_col.append(d_len + i)
                    new_data.append(cooc)
            

        # Modifying dictionary to store sentences
        # (we modify the dictionary later because we are reading it in the previous loop)
        for i in xrange(length):
            self.dictionary['SENT_{}'.format(i)] = d_len + i

        new_row = np.append(self.matrix.row,new_row)
        new_col = np.append(self.matrix.col,new_col)
        new_data = np.append(self.matrix.data,new_data)
        self.matrix = sp.coo_matrix((new_data, (new_row,new_col)),
            shape = (shape,shape),
            dtype = np.float64)
        """
        print self.matrix.shape, "New shape"

    def save(self, filename):
        
        with open(filename, 'wb') as savefile:
            pickle.dump((self.dictionary, self.matrix),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):

        instance = cls()

        with open(filename, 'rb') as savefile:
            instance.dictionary, instance.matrix = pickle.load(savefile)

        return instance
