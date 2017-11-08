from vv_align import VectorAlignment as VA
from get_corpus import get_corpus
get_corpus('text8', 'corpus1')
a = VA('corpus1', 'vectors1')
a.skip_generate_vectors()
#a.generate_vectors()
a.load_vectors()

get_corpus('cfilt_hin_corp_unicode', 'corpus2')
b = VA('corpus2', 'vectors2')
b.skip_generate_vectors()
#b.generate_vectors('word2vec')
b.load_vectors()

aligned_words = [line.strip().split() for line in open("files/parallel_hin_eng", 'r', encoding='utf-8').readlines()]

Ab = b.align_vectors(a, aligned_words[:-10])
b.compare_aligned(a, Ab, aligned_words[-10:])
#Ab = a.align_vectors(a, a.words[:200])
#a.compare_aligned(a, Ab, a.words[200:400])

