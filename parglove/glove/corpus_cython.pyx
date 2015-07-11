#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#distutils: language = c++

import numpy as np
import scipy.sparse as sp

from cpython cimport array as c_array
from array import array

from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_max(int a, int b): return a if a > b else b


cdef extern from "math.h":
    double c_abs "fabs"(double)


cdef class Matrix:
    """
    A sparse co-occurrence matrix storing
    its data as a vector of maps.
    """

    cdef int max_map_size
    cdef vector[unordered_map[int, float]] rows

    cdef vector[vector[int]] row_indices
    cdef vector[vector[float]] row_data

    def __cinit__(self, int max_map_size):

        self.max_map_size = max_map_size
        self.rows = vector[unordered_map[int, float]]()

        self.row_indices = vector[vector[int]]()
        self.row_data = vector[vector[float]]()

    cdef void compactify_row(self, int row):
        """
        Move a row from a map to more efficient
        vector storage.
        """

        cdef int i, col
        cdef int row_length = self.row_indices[row].size()

        cdef pair[int, float] row_entry
        cdef unordered_map[int, float].iterator row_iterator

        row_unordered_map = self.rows[row]

        # Go through the elements already in vector storage
        # and update them with the contents of a map, removing
        # map elements as they are transferred.
        for i in range(row_length):
            col = self.row_indices[row][i]
            if self.rows[row].find(col) != self.rows[row].end():

                self.row_data[row][i] += self.rows[row][col]
                self.rows[row].erase(col)

        # Resize the vectors to accommodate new
        # columns from the map.
        row_length = self.row_indices[row].size()
        self.row_indices[row].resize(row_length)
        self.row_data[row].resize(row_length)

        # Add any new columns to the vector.
        row_iterator = self.rows[row].begin()
        while row_iterator != self.rows[row].end():
            row_entry = deref(row_iterator)
            self.row_indices[row].push_back(row_entry.first)
            self.row_data[row].push_back(row_entry.second)
            inc(row_iterator)

        self.rows[row].clear()

    cdef void add_row(self):
        """
        Add a new row to the matrix.
        """

        cdef unordered_map[int, float] row_map

        row_map = unordered_map[int, float]()

        self.rows.push_back(row_map)
        self.row_indices.push_back(vector[int]())
        self.row_data.push_back(vector[float]())

    cdef void increment(self, int row, int col, float value):
        """
        Increment the value at (row, col) by value.
        """

        cdef float current_value

        while row >= self.rows.size():
            self.add_row()        

        self.rows[row][col] += value

        if self.rows[row].size() > self.max_map_size:
            self.compactify_row(row)

    cdef int size(self):
        """
        Get number of nonzero entries.
        """

        cdef int i
        cdef int size = 0

        for i in range(self.rows.size()):
            size += self.rows[i].size()
            size += self.row_indices[i].size()

        return size

    cpdef to_coo(self, int shape):
        """
        Convert to a shape by shape COO matrix.
        """

        cdef int i, j
        cdef int row
        cdef int col
        cdef int rows = self.rows.size()
        cdef int no_collocations

        # Transform all row maps to row arrays.
        for i in range(rows):
            self.compactify_row(i)

        no_collocations = self.size()

        # Create the constituent numpy arrays.
        row_np = np.empty(no_collocations, dtype=np.int32)
        col_np = np.empty(no_collocations, dtype=np.int32)
        data_np = np.empty(no_collocations, dtype=np.float64)
        cdef int[:,] row_view = row_np
        cdef int[:,] col_view = col_np
        cdef double[:,] data_view = data_np

        j = 0

        for row in range(rows):
            for i in range(self.row_indices[row].size()):

                row_view[j] = row
                col_view[j] = self.row_indices[row][i]
                data_view[j] = self.row_data[row][i]

                j += 1

        # Create and return the matrix.
        return sp.coo_matrix((data_np, (row_np, col_np)),
                             shape=(shape,
                                    shape),
                             dtype=np.float64)

    def __dealloc__(self):

        self.rows.clear()
        self.row_indices.clear()
        self.row_data.clear()


cdef inline int words_to_ids(list words, vector[int]& word_ids,
                      dictionary, int supplied, int ignore_missing):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied, a word is missing from it,
    and we are not ignoring out-of-vocabulary (OOV) words, an
    error value of -1 is returned.

    If we have an OOV word and we do want to ignore them, we use
    a -1 placeholder for it in the word_ids vector to preserve
    correct context windows (otherwise words that are far apart
    with the full vocabulary could become close together with a
    filtered vocabulary).
    """

    cdef int word_id

    word_ids.resize(0)

    if supplied == 1:
        for word in words:
            # Raise an error if the word
            # is missing from the supplied
            # dictionary.
            word_id = dictionary.get(word, -1)
            if word_id == -1 and ignore_missing == 0:
                return -1

            word_ids.push_back(word_id)

    else:
        for word in words:
            word_id = dictionary.setdefault(word,
                                            len(dictionary))
            word_ids.push_back(word_id)

    return 0

            
def construct_cooccurrence_matrix(corpus, dictionary, int supplied,
                                  int window_size, int ignore_missing,
                                  int max_map_size):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # Declare the cooccurrence map
    cdef Matrix matrix = Matrix(max_map_size)
    unigram_freq = {}
    X_values = {}

    # String processing variables.
    cdef list words
    cdef int i, j, outer_word, inner_word
    cdef int wordslen, window_stop, error
    cdef vector[int] word_ids

    # Pre-allocate some reasonable size
    # for the word ids vector.
    word_ids.reserve(1000)

    # Iterate over the corpus.
    for words in corpus:

        # Convert words to a numeric vector.
        error = words_to_ids(words, word_ids, dictionary, supplied, ignore_missing)
        if error == -1:
            raise KeyError('Word missing from dictionary')
        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            unigram_freq[outer_word] = unigram_freq.get(outer_word,0) + 1

            # Continue if we have an OOD token.
            if outer_word == -1:
                continue

            window_stop = int_min(i + window_size + 1, wordslen)

            for j in range(i, window_stop):
                inner_word = word_ids[j]

                if inner_word == -1:
                    continue

                # Do nothing if the words are the same.
                if inner_word == outer_word:
                    continue

                if inner_word < outer_word:
                    matrix.increment(inner_word,
                                     outer_word,
                                     1.0 / (j - i))
                    X_values[inner_word] = X_values.get(inner_word,0) + (1.0/ (j - i))
                    X_values[outer_word] = X_values.get(outer_word,0) + (1.0/ (j - i))
                else:
                    matrix.increment(outer_word,
                                     inner_word,
                                     1.0 / (j - i))
                    X_values[inner_word] = X_values.get(inner_word,0) + (1.0/ (j - i))
                    X_values[outer_word] = X_values.get(outer_word,0) + (1.0/ (j - i))
    
    # Create the matrix.
    mat = matrix.to_coo(len(dictionary))

    return mat, unigram_freq, X_values

cimport numpy as c_np
cdef vector[int] array_to_vector_i(c_np.ndarray[int, ndim=1] array):
    cdef long size = array.size
    cdef vector[int] vec
    cdef long i
    for i in range(size):
        vec.push_back(array[i])
    return vec

cdef vector[double] array_to_vector_d(c_np.ndarray[double, ndim=1] array):
    cdef long size = array.size
    cdef vector[double] vec
    cdef long i
    for i in range(size):
        vec.push_back(array[i])
    return vec


def add_sentences(matrix, dictionary, unigrams, X_vals, sentences, int length):
    """
    """
    csc_matrix = matrix.tolil()

    cdef vector[int] old_row = array_to_vector_i(matrix.row)
    cdef vector[int] old_col = array_to_vector_i(matrix.col)
    cdef vector[double] old_data = array_to_vector_d(matrix.data)
    

    cdef int d_len = len(dictionary)
    cdef shape = d_len + length
    cdef vector[int] new_row, new_col
    cdef vector[double] new_data

    cdef int num_new = 0
    cdef int d_id, sw_id

    # Define varibles for algorithm
    cdef double cooc
    cdef double factor
    cdef int tot_counts = sum(unigrams.values())
    cdef int tot_X = sum(X_vals.values())
    cdef int num_s_words
    
    cdef int i
    sentences_saved = []
    for d_i, d_word in enumerate(dictionary):
        if d_i % int(d_len/20) == 0: print d_i
        d_id = dictionary[d_word]
        for i,sentence in enumerate(sentences):
            sentence = list(sentence)
            sentences_saved.append(sentence)

            cooc = 0.0
            # Algorithm Starts
            
            '''
            # New attempt
            cooc = 0.0
            l = len(sentence)
            for i,s_word in enumerate(sentence):
                sw_id = dictionary.get(s_word,-1)
                if sw_id == -1:
                    continue
                cooc += csc_matrix[min(sw_id,d_id),max(sw_id,d_id)] / (l - i)
            #'''

            '''
            # Bigram based
            cooc = 0.0
            for i in range(len(sentence)-1):
                sw_id = dictionary.get(sentence[i], -1)
                if sw_id == -1:
                    continue
                if cooc == 0.0:
                    cooc = csc_matrix[min(d_id, sw_id), max(d_id, sw_id)]
                sw_id_2 = dictionary.get(sentence[i+1], -1)
                if sw_id_2 == -1:
                    continue
                cooc *= csc_matrix[min(sw_id, sw_id_2), max(sw_id, sw_id_2)] / X_vals[sw_id]
            cooc *= X_vals[d_id]
            #'''

            #'''
            # avg ij
            cooc = 0.0
            num_s_words = 0
            for s_word in set(sentence):
                sw_id = dictionary.get(s_word, -1)
                if sw_id == -1 or d_id == sw_id:
                    continue
                else:
                    factor = 1.0 * (tot_counts - unigrams[sw_id]) / tot_counts
                    #factor = 1.0
                    num_s_words += 1
                    cooc += csc_matrix[min(sw_id, d_id), max(sw_id, d_id)] * factor
            if num_s_words > 0:
                cooc /= num_s_words
            cooc *= unigrams[d_id]
            #'''

            '''
            # Prod bigrams
            cooc = unigrams[d_id]
            for s_word in set(sentence):
                sw_id = dictionary.get(s_word, -1)
                if sw_id == -1 or d_id == sw_id:
                    continue
                else:
                    cooc *= csc_matrix[min(sw_id, d_id), max(sw_id, d_id)] 
                    #cooc *= unigrams[sw_id]
                    #cooc *= X_vals[sw_id]
            #'''

            # Algorithm Ends
            if cooc != 0.0:
                new_row.push_back(d_id)
                new_col.push_back(d_len + i)
                new_data.push_back(cooc)
                num_new += 1
            # We have written Word-Sentence cooccurrence counts

    for i in range(num_new):
        old_row.push_back(new_row[i])
        old_col.push_back(new_col[i])
        old_data.push_back(new_data[i])
 
    row_np = np.empty(old_row.size(), dtype = np.int32)
    col_np = np.empty(old_col.size(), dtype = np.int32)
    data_np = np.empty(old_data.size(), dtype = np.float64)
    cdef int[:,] row_view = row_np
    cdef int[:,] col_view = col_np
    cdef double[:,] data_view = data_np
    
    for i in range(old_row.size()):
        row_view[i] = old_row[i]
        col_view[i] = old_col[i]
        data_view[i] = old_data[i]

    matrix = sp.coo_matrix((data_np, (row_np, col_np)),
            		 shape = (shape, shape),
            		 dtype = np.float64)
    csc_matrix = matrix.tolil()

    # Reset for S-S
    num_new = 0
    new_row.clear()
    new_col.clear()
    new_data.clear()
    old_row = array_to_vector_i(matrix.row)
    old_col = array_to_vector_i(matrix.col)
    old_data = array_to_vector_d(matrix.data)

    # Writing Sentence-Sentence cooccurrence counts:
    s_len = len(sentences_saved)
    for i,sentence_i in enumerate(sentences_saved):
        if i % int(s_len/20) == 0: print i
        for j in range(i):
            sentence_j = sentences_saved[j]
            cooc = 0.0
            # Algorithm Begins

            #'''
            # Set-Overlap * avg(unigram)
            cooc = 1.0 * len(set(sentence_i) & set(sentence_j)) * tot_counts / d_len
            #'''

            '''
            # sum over Set-overlap unigram of every word
            cooc = 0.0
            num_s_words = 0
            for word in set(sentence_i):
                if word in sentence_j:
                    sw_id = dictionary.get(word, -1)
                    if sw_id == -1:
                        cooc += tot_counts / d_len
                        num_s_words += 1
                    else:
                        cooc += unigrams[sw_id]
                        num_s_words += 1
            #'''

            '''
            # New variant - X_ij
            cooc = 0.0
            num_1_words = 0
            for word1 in sentence_i:
                id1 = dictionary.get(word1, -1)
                if id1 == -1: continue
                t_cooc = 0.0
                num_2_words = 0
                for word2 in sentence_j:
                    id2 = dictionary.get(word2, -1)
                    if id2 == -1: continue
                    t_cooc += csc_matrix[min(id1,id2),max(id1,id2)] * (1.0 - unigrams[id2]/tot_counts)
                    num_2_words += 1
                if num_2_words > 0:
                    cooc += unigrams[id1] * 1.0 * t_cooc / num_2_words
                    num_1_words += 1
            if num_1_words > 0:
                cooc /= num_1_words
            #'''

            '''
            # New variant - X_iA
            cooc = 0.0
            num_words = 0
            id2 = d_len + j
            for word1 in sentence_i:
                id1 = dictionary.get(word1, -1)
                if id1 == -1: continue
                cooc += csc_matrix[min(id1,id2),max(id1,id2)]
                num_words += 1
            if num_words > 0:
                cooc /= num_words
            #'''
            
            # Algorithm Ends
            if cooc != 0.0:
                new_row.push_back(d_len + j)
                new_col.push_back(d_len + i)
                new_data.push_back(cooc)
                num_new += 1
    
    for i in range(length):
        dictionary['SENT_{}'.format(i)] = d_len + i

    for i in range(num_new):
        old_row.push_back(new_row[i])
        old_col.push_back(new_col[i])
        old_data.push_back(new_data[i])
    
    row_np = np.empty(old_row.size(), dtype = np.int32)
    col_np = np.empty(old_col.size(), dtype = np.int32)
    data_np = np.empty(old_data.size(), dtype = np.float64)
    #cdef int[:,] row_view = row_np
    #cdef int[:,] col_view = col_np
    #cdef double[:,] data_view = data_np
    row_view = row_np
    col_view = col_np
    data_view = data_np
    
    for i in range(old_row.size()):
        row_view[i] = old_row[i]
        col_view[i] = old_col[i]
        data_view[i] = old_data[i]

    return sp.coo_matrix((data_np, (row_np, col_np)),
            		 shape = (shape, shape),
            		 dtype = np.float64),dictionary
