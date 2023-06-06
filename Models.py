import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import tf_lower_and_split_punct

"""
The Encoder:
1.Takes a list of token IDs(from context_text_processor)
2.Look up an embedding vector for each token(Using a layers.Embedding)
3.Processes the embeddings into a new sequence(Bidirectional GRU)
4.Returns the processed sequence. This will be passed to the attention head
"""


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        # 定义文本的processor(Textvectorition)
        self.text_processor = text_processor
        # 获取词典大小
        self.vocab_size = text_processor.vocabulary_size()
        # embedding维度
        self.units = units

        # Embeddin层
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                                   mask_zero=True)

        # The RNN layer
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(
                units,
                return_sequences=True,
                # Glorot 均匀分布初始化器，也称为 Xavier 均匀分布初始化器。
                recurrent_initializer='glorot_uniform'
            )
        )

    def call(self, x):
        # 1. embeddinglayers look up the embedding layers
        x = self.embedding(x)

        # 2. The GRU processes the sequence of embeddings.
        x = self.rnn(x)

        # 3. Returns the new sequence of embeddings.
        return x

    def convert_input(self, texts):
        # tf.newaxis:会为输入的变量新增维度，自适应神经网络的要求维度
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        # 调用call
        context = self(context)
        return context


"""
⭐ Attention layer:
"""


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        # layerNormalization：对每一个样本进行归一化
        # 与之相对应的是，BatchNormalization：对同一个Batch多个样本的单个通道进行归一化
        self.layernorm = tf.keras.layers.LayerNormalization()
        # Add层：对同一纬度的不同向量进行加和(与concatenate不同)
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        # attn_output: (batch_size, query_dimension, value_dimension)
        # attn_scores: (batch_size, num_heads, query_elements, key_elements)
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,  # 默认key与value相同
            # attention scores 是转化为注意力权重之前的值
            return_attention_scores=True
        )
        # 取每个head在一个scores上的均值(此模型只有一个head)
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores

        # 类似于残差链接
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


"""
The decoder:is to generate predictions for the next token 
at each location in the target sequence.
1.It looks up embeddings for each token in the target sequence.
2.It uses an RNN to process the target sequence, and keep track of what 
 has generated so far.
3.It uses RNN output as the "query" to the attention layer, 
when attending to the encoder's output.
4.At each location in the output it predicts the next token.
解码器采用的并不是双向的RNN 而是单向RNN，因为要顺序生成文字
"""


class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()

        self.words_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]'
        )

        self.id_to_words = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True
        )
        self.start_token = self.words_to_id('[START]')
        self.end_token = self.words_to_id('[END]')
        self.units = units

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   units, mask_zero=True)

        # RNN
        self.rnn = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'
                                       )
        # attention
        self.attention = CrossAttention(units)
        # output layer
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        # Lookup the embeddings
        x = self.embedding(x)

        # Process the target sequence
        x, state = self.rnn(x, initial_state=state)

        # RNN output as the Query for the attention over the context
        # x:query, context:key
        # 计算每一个输出x与context的attention
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        # : (batch_size, 1(x_len), 256)
        # print("x: ", x.shape)

        #  generate the logits predcition
        logits = self.output_layer(x)

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        # rnn.get_initial_state:用Start初始化RNN
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_words(tokens)
        # 沿轴拼接字符串
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        # 调用call方法
        # state: RNN的初始状态
        logits, state = self(
            context, next_token,
            state=state,
            return_state=True
        )

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            # logits: (64,1,5000)
            logits = logits[:, -1, :] / temperature

            # logits:(batch_size, numsofclasses)
            # categorical:对logits的每行通过概率进行抽样
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an 'end_token', set it 'done'
        done = done | (next_token == self.end_token)
        # 序列长度可能不一: 对结束了的句子用0-padding进行填补
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state


"""
Combine all the components to a single model for training:
"""


class Translator(tf.keras.Model):
    def __init__(self, units,
                 context_text_processor,
                 target_text_processor):
        super().__init__()
        # Build encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        # TODO(b/250038731): remove this
        try:
            # Delete the keras mask, so keras doesn't scale the loss+accuracy.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def translate(self,
                  texts,
                  *,
                  max_length=50,
                  temperature=0.0):
        # Process the input texts
        context = self.encoder.convert_input(texts)
        batch_size = tf.shape(texts)[0]

        # Setup the loop inputs
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):
            # Generate the next token
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature)

            # Collect the generated tokens
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        tokens = tf.concat(tokens, axis=-1)
        # 在(batch, query_seq, key_seq)中的query_seq维度concat
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        print(self.last_attention_weights.shape)
        result = self.decoder.tokens_to_text(tokens)
        return result

    def plot_attention(self, text, **kwargs):
        assert isinstance(text, str)
        output = self.translate([text], **kwargs)
        output = output[0].numpy().decode()

        # Translate中的self.last_attention_weights:
        # (batch_size, value
        attention = self.last_attention_weights[0]

        context = tf_lower_and_split_punct(text)
        context = context.numpy().decode().split()

        output = tf_lower_and_split_punct(output)
        output = output.numpy().decode().split()[1:]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        ax.matshow(attention, cmap='viridis', vmin=0.0)

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + output, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.set_xlabel('Input text')
        ax.set_ylabel('Output text')

class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    # 只保留Translate方法
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)