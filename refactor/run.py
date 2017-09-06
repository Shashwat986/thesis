#!python
import subprocess, os

# Step 1
# Creation of word vectors from corpus

class VectorAlignment:
  def __init__(self, file_name = None, output_file_name = None, options = None):
    self.env = os.environ.copy()
    if file_name is None:
      print("Downloading test corpus")
      subprocess.call(['curl', '-O', 'http://mattmahoney.net/dc/text8.zip'])
      subprocess.call(['unzip', 'text8.zip'])
      subprocess.call(['rm', 'text8.zip'])
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

  def set_file_name(file_name):
    self.file_name = file_name

  def set_output_file_name(file_name):
    self.output_file_name = file_name

  def set_default_values(self):
    for key in ['VECTOR_SIZE', 'WINDOW_SIZE', 'VOCAB_MIN_COUNT']:
      if key in self.options:
        self.env[key] = str(self.options[key])

  def generate_vectors(self, method="glove"):
    if method not in ['glove', 'word2vec']:
      return None

    print("Using {} to generate word vectors.".format(method))

    self.set_default_values()

    exit_code = subprocess.call([
      method + '.sh',
      self.file_name,
      self.output_file_name
    ], env=self.env)

    if exit_code == 0:
      print("Vectors created: {}".format(self.output_file_name))
    else:
      raise Exception("Error in word vector generation")

  def align_vectors(self, other):
    pass

