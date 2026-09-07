import numpy as np
import tensorflow as tf

#define positional encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(position,
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    if tf.is_tensor(angle_rads):
        angle_rads = angle_rads.numpy()
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

#transformer implemintation
def scaled_dot_product_attention(q, k, v):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    seq_len = q.shape[1]

    pos   = positional_encoding(np.arange(seq_len)[:, np.newaxis], self.d_model)
    q = self.wq(q) + pos
    k = self.wk(k) + pos
    v = self.wv(v)  
    q = self.split_heads(q, batch_size) 
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size)  
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  

    output = self.dense(concat_attention)  

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'), 
      tf.keras.layers.Dense(d_model)  
  ])

# Encoder
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x):

    attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output)
    out1 = x + attn_output
    #out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    out2 = out1 + ffn_output
    #out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

# Decoder
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Dense(d_model, activation='relu')

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x):
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # (batch_size, input_seq_len, d_model)

# Define Model
inp = tf.keras.layers.Input(shape = (72, 1))
x = Encoder(1, d_model=64, num_heads=4, dff=1024, rate=0.1)(inp)
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(6, activation='softmax')(x)
Transformer = tf.keras.models.Model(inputs = inp, outputs = out)

inp = tf.keras.layers.Input(shape = (72, 1))
x = Encoder(1, d_model=128, num_heads=4, dff=1024, rate=0.1)(inp)
x = tf.keras.layers.Flatten()(x)

out = tf.keras.layers.Dense(6, activation='softmax')(x)
Client_Transformer = tf.keras.models.Model(inputs = inp, outputs = out)

'''
class Transformer(tf.keras.Model):
  
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
    self.flatt_layer = tf.keras.layers.Flatten()
    self.final_layer = tf.keras.layers.Dense(12, activation='softmax')
  
  def call(self, inp):
    enc_output = self.encoder(inp)  # (batch_size, inp_seq_len, d_model)
    final_output = self.flatt_layer(enc_output)
    final_output = self.final_layer(final_output)  # (batch_size, tar_seq_len, target_vocab_size)
    return final_output
'''
