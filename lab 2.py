import numpy as np
import pandas as pd


# 1. Preparação dos dados


vocab = {
    "o": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartao": 3
}

df_vocab = pd.DataFrame(list(vocab.items()), columns=["palavra", "id"])

frase = ["o", "banco", "bloqueou", "cartao"]
input_ids = [vocab[palavra] for palavra in frase]

vocab_size = len(vocab)
d_model = 64
embedding_table = np.random.randn(vocab_size, d_model)

X = embedding_table[input_ids]
X = X[np.newaxis, :, :]   # (batch, seq_len, d_model)


# 2. Funções auxiliares


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + eps)


# 3. Self-Attention


def self_attention(x):
    d_model = x.shape[-1]

    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)

    Q = x @ WQ
    K = x @ WK
    V = x @ WV

    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(d_model)

    att_weights = softmax(scores)
    output = att_weights @ V

    return output


# 4. Feed-Forward Network


def feed_forward(x):
    d_model = x.shape[-1]
    d_ff = 256

    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)

    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)   # ReLU
    output = hidden @ W2 + b2

    return output



# 5. Encoder Layer


def encoder_layer(x):
    x_att = self_attention(x)
    x_norm1 = layer_norm(x + x_att)

    x_ffn = feed_forward(x_norm1)
    x_out = layer_norm(x_norm1 + x_ffn)

    return x_out



# 6. Transformer Encoder


def transformer_encoder(x, n_camadas=6):
    for _ in range(n_camadas):
        x = encoder_layer(x)
    return x



# 7. Execução


Z = transformer_encoder(X)

print("Shape de entrada:", X.shape)
print("Shape final:", Z.shape)
print("\nSaída do encoder (Z):")
print(Z)