Name: Harshavardhini Katta Student Number: 700778822

A) The block initializes two main components:
Multi-Head Self-Attention (nn.MultiheadAttention) using d_model=128 and num_heads=8.

B) Residual connections are applied after attention and FFN 
x + attn_out and x + ffn_out help preserve original information and make training more stable.

C) This tests whether the model processes a batch of 32 sequences, each with 10 tokens of 128 dimensions.
