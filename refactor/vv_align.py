#!python
import subprocess, os, gc
import numpy as np
from scipy.spatial.distance import cosine, euclidean

VECTOR_STATE = {
  'created' : 10,
  'loaded'  : 20
}

class VectorAlignment:
  def __init__(self, file_name = None, output_file_name = None, options = None):
    self.env = os.environ.copy()
    if file_name is None:
      print("Downloading test corpus")
      subprocess.call(['curl', '-O', 'http://mattmahoney.net/dc/text8.zip'])
      subprocess.call(['unzip', 'text8.zip'])
      subprocess.call(['rm', 'text8.zip'])
      subprocess.call(['mkdir', '-p', './files'])
      subprocess.call(['mv', 'text8', './files/text8'])
      self.file_name = 'text8'
    else:
      self.file_name = file_name

    if output_file_name is None:
      self.output_file_name = 'vectors'
    else:
      self.output_file_name = output_file_name

    if options is None:
      self.options = {}
    else:
      self.options = options

    self.vector_state = None
    self.vectors = {}

  '''
    Helper methods to keep track of the current state of the vectors
  '''
  def set_vector_state(self, state):
    if state not in VECTOR_STATE: return False
    self.vector_state = VECTOR_STATE[state]

  def vectors_(self, state):
    if state not in VECTOR_STATE: return False
    return (self.vector_state >= VECTOR_STATE[state])

  '''
    Setting default values for Vector generation
  '''
  def set_default_values(self):
    self.env['VECTOR_SIZE'] = '50'
    self.env['WINDOW_SIZE'] = '15'
    self.env['VOCAB_MIN_COUNT'] = '5'

    for key in ['VECTOR_SIZE', 'WINDOW_SIZE', 'VOCAB_MIN_COUNT']:
      if key in self.options:
        self.env[key] = str(self.options[key])

    self.vector_size = int(self.env['VECTOR_SIZE'])

  '''
    Vector Generation
  '''
  def generate_vectors(self, method="glove"):
    if method not in ['glove', 'word2vec']:
      return None

    print("Using {} to generate word vectors.".format(method))

    self.set_default_values()

    exit_code = subprocess.call([
      './' + method + '.sh',
      self.file_name,
      self.output_file_name
    ], env=self.env)

    if exit_code == 0:
      self.set_vector_state('created')
      print("Vectors created: {}".format(self.output_file_name))
    else:
      raise Exception("Error in word vector generation")

  '''
    If we already have the file containing vectors, we can load it directly for the next steps
  '''
  def skip_generate_vectors(self, output_file_name = None):
    self.set_default_values()
    if output_file_name is not None: self.output_file_name = output_file_name
    self.set_vector_state('created')

  '''
    Loading the vectors in memory from the output_file_name
  '''
  def load_vectors(self):
    print("Opening file")
    f = open('./files/' + self.output_file_name, encoding='utf-8')
    print("Loading file line-by-line.")

    lines, dim = list(map(int,f.readline().strip().split()))
    print("\nTotal Lines: ", lines)
    self.words = [None] * lines

    if dim != self.vector_size:
      self.vector_size = dim
      print("Changing vector_size based on what's written in vectors file: ", dim)

    i = 0
    for line in f:
      split_line = line.strip().split()
      word = "".join(split_line[:-self.vector_size])
      vector = np.fromiter(map(float, split_line[-self.vector_size:]), np.float)

      if len(vector) != self.vector_size:
        print("Error in line length. Ignoring the word " + word)
        continue

      self.vectors[word] = vector
      try:
        self.words[i] = word
      except IndexError:
        print("Some issue with number of lines.")
        self.words.append(word)

      i += 1
      if i % 1000 == 0: print("{} lines processed".format(i), end="\r")

    print("\nLoading done.")
    f.close()
    self.set_vector_state('loaded')

  def get_vector(self, key, Ab = None):
    if not (self.vectors_('loaded')):
      print("Vectors not loaded");

    if key not in self.vectors:
      return None

    if Ab is None:
      return self.vectors[key]
    else:
      if Ab.shape[0] != self.vector_size + 1:
        print("Error in Transformation Matrix dimensions. Do not match vector dimensions")
        return None

      vector = self.vectors[key]
      return np.delete(
        np.dot(
          Ab, np.append(vector, 1)
        ).reshape((self.vector_size + 1, 1))
        , -1
      )

  '''
    Aligning vectors for two objects
  '''
  def align_vectors(self, other, aligned_words = []):
    if not (self.vectors_('created') and other.vectors_('created')):
      print("Create vectors first");
      return False

    if not (self.vectors_('loaded')): self.load_vectors()
    if not (other.vectors_('loaded')): other.load_vectors()

    if self.vector_size != other.vector_size:
      print("Vector sizes don't match")
      return False

    if len(aligned_words) <= self.vector_size:
      print("Number of aligned words is too low for a solution")
      return False

    def get_lines():
      for word in aligned_words:
        word_self = None
        word_other = None
        if type(word) == type([]):
          if len(word) > 1:
            word_self = word[0]
            word_other = word[1]
          else:
            word_self = word[0]
            word_other = word[0]
        else:
          word_self = word
          word_other = word

        if word_self in self.vectors and word_other in other.vectors:
          yield word, self.vectors[word_self], other.vectors[word_other]

    dimensions = self.vector_size
    aligned_words_count = len(aligned_words)

    Y = np.zeros((dimensions + 1, aligned_words_count))
    X = np.zeros((dimensions + 1, aligned_words_count))

    print("Creating matrix")
    generator = get_lines()
    for n in range(aligned_words_count):
      word, vector_self, vector_other = next(generator)
      Y[:, n] = np.append(vector_other, 1)
      X[:, n] = np.append(vector_self, 1)

    print("Matrix Created. Solving")
    gc.collect()

    Ab = np.dot(Y, np.linalg.pinv(X))

    # TODO: Save the matrix in a save file

    if not np.any(Ab):
      print("There has been some issue with the matrix creation")

    print(Ab.shape)
    print(np.allclose(np.dot(Ab, X), Y))

    return Ab

  def compare_aligned(self, other, Ab, aligned_words):
    cosine_difference = 0.0
    euclidean_difference = 0.0
    count_words = 0
    for word in aligned_words:
      word_self = None
      word_other = None
      if type(word) == type([]):
        if len(word) > 1:
          word_self = word[0]
          word_other = word[1]
        else:
          word_self = word[0]
          word_other = word[0]
      else:
        word_self = word
        word_other = word

      aligned_self = self.get_vector(word_self, Ab)
      original_other = other.get_vector(word_other)

      if aligned_self is None or original_other is None:
        continue

      original_other_t = original_other.reshape((other.vector_size, 1))

      count_words += 1
      cosine_difference += cosine(aligned_self, original_other_t)
      euclidean_difference += euclidean(aligned_self, original_other_t)

    cosine_difference /= count_words
    euclidean_difference /= count_words

    print("Cosine difference: ", cosine_difference)
    print("Euclidean difference: ", euclidean_difference)
