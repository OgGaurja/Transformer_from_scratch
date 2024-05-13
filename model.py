# Model of the Transformer

# 1)Input Embedding
# For each token generates an Input ID (Position in Vocabulary)
# A vector generated called Embedding of the Input Id of size 512

import torch
import torch.nn as nn
import math


# Creating class of Input Embeddings
# inheriting from nn.Module
class InputEmbeddings(nn.Module):

    # Creating constructor
    # Taking size of embedding and size of vocabulary
    def __inti__(self, d_model: int, vocab_size: int):

        ## constructor of inherited class
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # learn OOps in PYthon
        self.embedding = nn.Embedding(vocab_size, d_model)

    ### forward method
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
 