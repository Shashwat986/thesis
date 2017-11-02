from vv_align import VectorAlignment as VA
a = VA('text8')
a.skip_generate_vectors()
a.load_vectors()

Ab = a.align_vectors(a, a.words[:200])
a.compare_aligned(a, Ab, a.words[200:400])

