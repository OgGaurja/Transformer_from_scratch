import torch
import torch.nn as nn
import math




class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0, "d_model is not divisible by 4"
        
        ## Integer division //
        ## d_k = dimension of each head
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # A static method can be called without an instance
    @staticmethod
    def attention(query, key, value,  mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch,h, Seq_len, d_k) x (Batch,h, d_k, Seq_len) = (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)

        ## before applying softmax, mask the values
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        # Now apply softmax
        attention_scores =  attention_scores.softmax(dim = -1)  #(Batch, h, seq_len,seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        #(Batch, h, seq_len,seq_len) x (Batch,h, Seq_len, d_k) = (Batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores    
    
        



    def forward(self, q,k,v, mask):

        print('FORWARD METHOD CALLED||FORWARD METHOD CALLED||FORWARD METHOD CALLED')

        ## final arrays for Multihead
        ## dimension of q = seq x d_model
        # w_q == > [d_mode, d_moel]
        # seq x d_model (MatMUl) ==>>  
        query  = self.w_q(q)   ## dimensions [seq x d_model]
        key  = self.w_k(k)   
        value  = self.w_v(v)

        ## dividing each matrix into number of heads

        #(Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch,h, Seq_len, d_k)
        query =  query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key =  key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value =  value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        ## Applying attention on each head matrix
        # (Batch, h, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key, value,mask,dropout= self.dropout)

        #(Batch, h, seq_len, d_k) --> (Batch,  seq_len, h ,d_k) using transpose --> (Batch, seq_len, h*d_k = d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        ## Linear transformation of value matrix
        return self.w_o(x)



print(f"Hello, and version of torch: {torch.__version__}")


