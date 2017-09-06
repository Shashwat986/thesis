#!python
import subprocess, os
current_env = os.environ.copy()

# Step 1
# Creation of word vectors from corpus

class VectorAlignment:
  def __init__(self, file_name = None, options = None):
    if file_name is None:
      print("Downloading test corpus")
      subprocess.call(['curl', '-O', 'http://mattmahoney.net/dc/text8.zip'])
      subprocess.call(['unzip', 'text8.zip'])
      subprocess.call(['rm', 'text8.zip'])
      self.file_name = 'text8'
    else:
      self.file_name = file_name

    if options is None:
      self.options = {}
    else:
      self.options = options

  def set_default_values(self):
    if self.options.get('d') is None:
      self.options['d'] = 200



  def generate_vectors(self, method="glove"):
    print("Using {} to generate word vectors.".format(method))

    # Setting default values for required params










