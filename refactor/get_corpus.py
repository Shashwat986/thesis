import subprocess, glob

def get_corpus(corpus_id, file_name='input'):
  if corpus_id == 'text8':
    subprocess.call(['curl', '-O', 'http://mattmahoney.net/dc/text8.zip'])
    subprocess.call(['unzip', 'text8.zip'])
    subprocess.call(['rm', 'text8.zip'])
    subprocess.call(['mkdir', '-p', './files'])
    subprocess.call(['mv', 'text8', file_name])
    self.file_name = 'text8'
  elif corpus_id == 'cfilt_hin_corp_unicode':
    subprocess.call(['mkdir', '-p', './temp_files'])
    subprocess.call(['curl', '-O', 'http://www.cfilt.iitb.ac.in/hin_corp_unicode.tar'])
    subprocess.call(['tar', '-xvf', 'hin_corp_unicode.tar', '-C', 'temp_files'])
    subprocess.call(['rm', 'hin_corp_unicode.tar'])
    with open(file_name, 'w', encoding='utf-8') as f:
      subprocess.call(['cat'] + glob.glob('temp_files/hin_corp_unicode/*.txt'), stdout=f)
    subprocess.call(['rm', '-r', 'temp_files/hin_corp_unicode'])
  else:
    return None

  return True
