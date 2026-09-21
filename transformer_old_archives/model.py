import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_modelo, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_modelo % num_heads == 0, "d_modelo deve ser divisível por num_heads"
        """
        O modelo deve receber um número de heads que possa dividir a dimensão do modelo,
        caso contrário, aconteceriam alguns problemas bem tenebrosos lá pra frente
        """

        self.d_modelo = d_modelo
        self.num_heads = num_heads
        self.d_k = d_modelo // num_heads
        
        self.W_Q = nn.Linear(d_modelo, d_modelo)
        self.W_K = nn.Linear(d_modelo, d_modelo)
        self.W_V = nn.Linear(d_modelo, d_modelo)
        self.W_O = nn.Linear(d_modelo, d_modelo)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        atencao_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            atencao_scores = atencao_scores.masked_fill(mask == 0, -1e9)
        atencao_prob = torch.softmax(atencao_scores, dim=-1)
        saida = torch.matmul(atencao_prob, V)
        return saida
        
    def dividir_heads(self, x):
        batch_size, tamanho_seq, d_modelo = x.size()
        return x.view(batch_size, tamanho_seq, self.num_heads, self.d_k).transpose(1, 2)
        
    def juntar_heads(self, x):
        batch_size, _, tamanho_seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, tamanho_seq, self.d_modelo)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.dividir_heads(self.W_Q(Q))
        K = self.dividir_heads(self.W_K(K))
        V = self.dividir_heads(self.W_V(V))
        
        atencao_saida = self.scaled_dot_product_attention(Q, K, V, mask)
        saida = self.W_O(self.juntar_heads(atencao_saida))
        return saida

class FeedForward(nn.Module):
    def __init__(self, d_modelo, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_modelo, d_ff)
        self.fc2 = nn.Linear(d_ff, d_modelo)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncodingPosicional(nn.Module):
    def __init__(self, d_modelo, max_tamanho_seq):
        super(EncodingPosicional, self).__init__()
        
        pos = torch.zeros(max_tamanho_seq, d_modelo)
        posicao = torch.arange(0, max_tamanho_seq, dtype=torch.float).unsqueeze(1)
        termo_div = torch.exp(torch.arange(0, d_modelo, 2).float() * -(np.log(10000.0) / d_modelo))
        
        pos[:, 0::2] = torch.sin(posicao * termo_div)
        pos[:, 1::2] = torch.cos(posicao * termo_div)
        
        self.register_buffer('pos', pos.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pos[:, :x.size(1)]

class CamadaEncoder(nn.Module):
    def __init__(self, d_modelo, num_heads, d_ff, dropout):
        super(CamadaEncoder, self).__init__()
        self.autoatencao = MultiHeadAttention(d_modelo, num_heads)
        self.feed_forward = FeedForward(d_modelo, d_ff)
        self.norm1 = nn.LayerNorm(d_modelo)
        self.norm2 = nn.LayerNorm(d_modelo)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        atencao_saida = self.autoatencao(x, x, x, mask)
        x = self.norm1(x + self.dropout(atencao_saida))
        saida_ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(saida_ff))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, tamanho_vocab, d_modelo, num_heads, d_ff, num_camadas, dropout, num_classes, max_tamanho_seq):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(tamanho_vocab, d_modelo)
        self.pos_encoder = EncodingPosicional(d_modelo, max_tamanho_seq)
        
        self.camadas = nn.ModuleList([
            CamadaEncoder(d_modelo, num_heads, d_ff, dropout) for _ in range(num_camadas)
        ])
        
        self.fc_out = nn.Linear(d_modelo, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        for camada in self.camadas:
            x = camada(x, mask)
            
        x_cls = x[:, 0, :]
        
        saida = self.fc_out(self.dropout(x_cls))
        return saida