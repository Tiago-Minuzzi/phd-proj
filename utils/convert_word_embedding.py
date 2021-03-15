import sys
import pandas as pd
from keras.preprocessing.text import Tokenizer

TABELA = sys.argv[1]
SEQCOL = 'sequence'
LABCOL = 'class'

# Declare tokenizer
tkz_seq = Tokenizer(num_words = None, split = ' ', char_level = True, lower = True)
tkz_lab = Tokenizer()

# Read file
df = pd.read_csv(TABELA)
sequencias = df[SEQCOL]
categorias = df[LABCOL]

# Tokenize sequences
tkz_seq.fit_on_texts(sequencias)
x_seq_arrays = tkz_seq.texts_to_sequences(sequencias)
vocab_size_seq = len(tkz_seq.word_index) + 1

#print(tkz_seq.word_counts)
#print(tkz_seq.word_index)
#print(tkz_seq.word_docs)
#print(tkz_seq.document_count)
#print(vocab_size_seq)

# Tokenize labels
tkz_lab.fit_on_texts(categorias)
toklabs = tkz_lab.texts_to_sequences(categorias)
vocab_size_lab = len(tkz_lab.word_index) + 1

#print(tkz_lab.word_counts)
#print(tkz_lab.word_index)
#print(tkz_lab.word_docs)
#print(tkz_lab.document_count)
#print(vocab_size_lab)
