### 基于Attention机制的机器翻译模型

> 来源：https://tensorflow.google.cn/text/tutorials/nmt_with_attention

#### 一、引言

##### 1.1 项目简介

​	本项目是来自Tensorflow官网的示例项目，代码简介，原理清晰。因此选择这个项目作为Tensorflow和深度学习的练手学习项目。此项目通过 Seq2Seq 模型，实现了Spanish-English的翻译任务。项目主要基于[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025v5) (Luong et al., 2015).

​	全部代码见文章末尾。

##### 1.2 模型简介

​	Attention机制是当下最流行的深度学习机制之一，突破了传统CNN、RNN在并行训练、长序列学习上的壁垒，以Attention机制为基础的Transformer模型更是开创了NLP的新纪元，诞生了GPT、Bert等模型。

​	本项目也同样是基于Attention机制，同时引入了RNN获取时序信息，以实现文本翻译。

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230404150336620.png" alt="image-20230404150336620" style="zoom: 80%;" />

- ***在本文中的概念注解如下：***

  1. context：输入待翻译的tokens——[Todavía, está, en, casa]
  2. target_in：翻译序列的参考样本(上文)，其实就是翻译结果前移一位——[[START], Are, you, still, at, home]
  3. target_out：翻译结果，即模型输出的labels。target_in向后平移1位——[Are, you, still, at, home, [END]]；

  其中，(context, target_in)作为 模型的输入，target_out 作为模型的输出。

- **Encoder与Decoder：**

  1. **Encoder：**包含 Embedding 层与 RNN 层

     a. Embedding层：将 tokens 转化为 tensor；

     b. RNN层：获取context的时序信息，输出每个时间节点的信息。

  2. **Decoder:** 包含 Embedding 层、Attention 层、RNN 层

     a. Embedding层：将tokens转为tensor；

     b. RNN 层 ：获取生成文本的时序信息（翻译新的词时需要获取上文的信息）

     c. Attention层：根据RNN的时序信息，计算每一个时间节点rnn的输出与context之间的attention值

     d. output_layer：根据attention值，通过全连接层输出最终的logits(维度为vocab_size)，即最终转化为了多分类任务

#### 二、数据部分

##### 2.1 数据获取

​	本数据是来自 Tensorflow 的开源数据，使用tf.keras.utils.get_file()方法获取

```python
# Download the file
import pathlib

path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
```

##### 2.2 数据预处理

- 数据预处理主要分为5个步骤：

  1. 数据的预览与核验
  2. 数据读取与切分

  2. 数据清洗与处理（去除标点、增加[START], [END]token）
  3. 创建 word2id 字典以及 id2word 字典
  4. 将所有句子填充为相同的长度(Padding)

###### 2.2.1 数据预览

​	数据每行由English - Spanish对组成

```python
text = path_to_file.read_text(encoding='utf-8')
lines = text.splitlines()
pairs = [line.split('\t') for line in lines]
pairs[1000:1005]
```

```python
>>[['He is a DJ.', 'Él es DJ.'],
 ['He is here!', '¡Él está aquí!'],
 ['He is kind.', 'Él es gentil.'],
 ['He is kind.', 'Él es amable.'],
 ['He is kind.', 'Él es generoso.']]
```

###### 2.2.2 数据读取与切分

​	定义数据读取方法，返回 target(翻译结果) 与 context(待翻译文本)

```python
# read the raw data, return english(target) and spanish(context)
def load_data(path):
    text = path.read_text(encoding='utf-8')
    # ⭐
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    
    context = np.array([context for target,context in pairs])
    target = np.array([target for target, context in pairs])
    
    return target, context

target_raw, context_raw =load_data(path_to_file)
print('Spanish:', context_raw[-1])
print('English:', target_raw[-1])
```

​	在读取数据后，对数据集进行切分，分别得到训练数据集与测试数据集。这些数据集在之后还需要进行处理。

```python
# create a tf.data.dataSet of strings that shuffles and batches them efficiently
BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64

# ⭐ 随机切分训练、验证集
is_train = np.random.uniform(size=(len(target_raw))) < 0.8

train_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
    .shuffle((BUFFER_SIZE))
    .batch(BATCH_SIZE)
)
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
```

​	让我们来看看切分后的数据：

```python
for example_context_strings, example_target_strings in train_raw.take(1):
    print(example_context_strings[:2])
    print()
    print(example_target_strings[:2])
```

```python
>>tf.Tensor(
    [b'Tom fue muy amable con Mary.'
     b'Hay una necesidad urgente de m\xc3\xa1s personas que donen su tiempo y su dinero.'], shape=(2,), dtype=string)

tf.Tensor(
    [b'Tom was very kind to Mary.'
     b'There is an urgent need for more people to donate their time and money.'], shape=(2,), dtype=string)
```

###### 2.2.3 数据清洗与处理

​	在西班牙语中，具有大量的标点符号和语态语调如"¿íá?"等，在实际的训练中，我们需要对其进行处理转化为无标点的tokens

```python
def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text
```

样本处理效果：

```python
example_text = tf.constant('¿Todavía está en casa?')
print(example_text.numpy().decode()) # tensorflow中的unicode，需要进行解码才能显示
print(tf_lower_and_split_punct(example_text).numpy().decode())
```

```python
>> ¿Todavía está en casa?
>> [START] ¿ todavia esta en casa ? [END] 
```

###### 2.2.4 创建字典与填充

​	我们使用tf.keras.layers.TextVectorization()，此方法可以自动根据数据集创建数据字典，我们可以传入数据预处理的方法。并且TextVectorization()为我们提供了word2id的调用方法。

- 创建Spanish字典

```python
# 词典大小
max_vocab_size = 5000

# 定义 Spainish TextVectorization
context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)
# 调用adapt方法，构建字典
context_text_processor.adapt(train_raw.map(lambda context, target: context))
# 获取字典前十个word
context_text_processor.get_vocabulary()[:10]
```

```python
>> ['', '[UNK]', '[START]', '[END]', '.', 'que', 'de', 'el', 'a', 'no']
```

- 创建English字典

```python
# English TextVectorization
target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)
target_text_processor.adapt(train_raw.map(lambda context, target: target))
target_text_processor.get_vocabulary()[:10]
```

```python
>> ['', '[UNK]', '[START]', '[END]', '.', 'the', 'i', 'to', 'you', 'tom']
```

- id2word字典

  要构建id2word字典，只需要获取 TextVectorization 的 vocabulary即可（行号-id）

```python
context_vocab = np.array(context_text_processor.get_vocabulary())
target_vocab = np.array(target_text_processor.get_vocabulary())
```

​	测试一下输出：

```python
example_tokens = context_text_processor(example_context_strings)
print(example_tokens[0].numpy())
tokens = context_vocab[example_tokens[0].numpy()]
print(' '.join(tokens))
```

```python
>> 	[   2 2167    6   11  256    4    3]
	[START] salga de la clase . [END]
```

- 查看数据的大致信息：

```python
# 获得zero-padded 形状
plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens.to_tensor())
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens.to_tensor() != 0)
plt.title('Mask')
```

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230404174909139.png" alt="image-20230404174909139" style="zoom:80%;" />

###### 2.2.5 输入转化

​	输入到模型中的数据应为(context, target_in), (target_out)，分别作为inputs和labels，输入输出的内容已在1.2介绍

```python
"""
将 context, target转化为模型所需要得输入：
inputs: (context, target_in)
outputs: target_out
其中target_in 与 target_out刚好错位，target_out中的元素对应着target_in的下一个元素
"""
def process_text(context, target):
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    target_in = target[:,:-1].to_tensor()
    target_out = target[:,1:].to_tensor()
    return (context, target_in), target_out

# 对Dataset类调用map方法即可对数据进行处理
train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)
```

​	查看一下输出：

```python
for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    print("context: ", ex_context_tok[0, :10].numpy())
    print("target_in: ", ex_tar_in[0, :10].numpy())
    print("target_out: ", ex_tar_out[0, :10].numpy())
```

```python
>> context:  [   2 1082   22    5   30  292    4    3    0    0]
>> target_in:  [  2 245  29   6 125   8   4   0   0   0]
>> target_out:  [245  29   6 125   8   4   3   0   0   0]
```

#### 三、Encoder与Decoder

##### 3.1 Encoder层

​	Encoder的作用是根据context生成相应的序列向量，这些向量包含了序列的语义信息以及时序信息。这里 Encoder 用了双向RNN。首先确定隐藏层大小 units 为256

```
UNITS = 256
```

​	Encoder的具体架构在先前已有介绍，Encoder 层最终返回的是context每个时间节点的时序信息。由于train_ds 和 val_ds 已经将 words 转化为了 ids，因此，在encoder的call()函数中并没有调用相应的TextVectorizaion：

```python
UNITS = 256
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
        #1. embedding layers look up the embedding layers
        x = self.embedding(x)
    
        #2. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        
        #3. Returns the new sequence of embeddings.
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
```

调用Encoder，使用context的TextVectorization进行初始化

```python
# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')
```

```
Context tokens, shape (batch, s): (64, 16)
Encoder output, shape (batch, s, units): (64, 16, 256)
```

##### 3.2 Attention层

​	Attention层本质上是Decoder的子层。实现了Decoder输出语义与Encoder语义信息的Attention信息的计算，从而使得Decoder生成文本时可以充分利用Encoder的语义、时序信息。如图所示，Query为每个时间节点Decoder的语义信息(包括了RNN、前一个token的语义)，Key和Value均为Encoder输出的信息（RNN在context上每一个时间节点的信息）

![image-20230405002613266](C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230405002613266.png)

```python
"""
⭐ Attention layer:
"""
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1,**kwargs)
        # layerNormalization：对每一个样本进行归一化
        # 与之相对应的是，BatchNormalization：对同一个Batch多个样本的单个通道进行归一化
        self.layernorm = tf.keras.layers.LayerNormalization()
        # Add层：对同一纬度的不同向量进行加和(与concatenate不同)
        self.add = tf.keras.layers.Add()
    
    def call(self, x, context):
        # attn_output: (batch_size, query_dimension, value_dimension)
        # attn_scores: (batch_size, num_heads, query_elements, key_elements)
        # att_scores是计算attention值之前的权重
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,    # 默认key与value相同
            # attention scores 是转化为注意力权重之前的值
            return_attention_scores=True
        )
        # 取每个head在一个scores上的均值(此模型只有一个head)
        attn_scores = tf.reduce_mean(attn_scores,axis=1)
        self.last_attention_weights = attn_scores
        
        # 类似于残差链接？
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x
```

   测试一下attention层，输入为ex_target_in的embedding以及之前Encoder输出的ex_context：

```python
attention_layer = CrossAttention(UNITS)

# Attend to the encoded tokens
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),
                                  output_dim=UNITS, mask_zero=True)
ex_tar_embed = embed(ex_tar_in)

result = attention_layer(ex_tar_embed, ex_context)

print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
print(f'Attention result, shape (batch, t, units): {result.shape}')
print(f'Attention weights, shape (batch, t, s):    {attention_layer.last_attention_weights.shape}')
```

```python
>> Context sequence, shape (batch, s, units): (64, 16, 256)
>> Target sequence, shape (batch, t, units): (64, 17, 256)
>> Attention result, shape (batch, t, units): (64, 17, 256)
>> Attention weights, shape (batch, t, s):    (64, 17, 16)
```

绘制出Attention计算过程中context的权重图

```python
attention_weights = attention_layer.last_attention_weights
mask=(ex_context_tok != 0).numpy()

plt.subplot(1, 2, 1)
plt.pcolormesh(mask*attention_weights[:, 0, :])
plt.title('Attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(mask)
plt.title('Mask');
```

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230405004847784.png" alt="image-20230405004847784" style="zoom:80%;" />

##### 3.3 Decoder层

​	Decoder 完成了对目标文本的生成任务，主要流程:

1. 为每个tokens生成相应的embedding；
2. 使用 RNN 获取已生成句子的时序、语义信息；
3. 将 RNN 输出作为attention层的query，计算与encoders的输出的attention值；
4. 对于每个已知时间节点的输出，生成下一个时间节点的词。

​	由于生成文本是单向的，所以Decoder中的 RNN 仅仅使用了单层网络

```python
"""
The decoder:is to generate predictions for the next token 
at each location in the target sequence.
1.It looks up embeddings for each token in the target sequence.
2.It uses an RNN to process the target sequence, and keep track of what 
t has generated so far.
3.It uses RNN output as the "query" to the attention layer, 
when attending to the encoder's output.
4.At each location in the output it predicts the next token.
解码器采用的并不是双向的RNN 而是单向RNN，因为要顺序生成文字
"""
class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        
        self.words_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]'
        )
        
        # 设置参数 invert=True 可以实现id2word
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
```

定义 call() 函数，接受三个参数。其中，

 1. inputs - context, x 

    - context：Encoder 的输出

    - x ：target_in

 2. state - 可选，前一个结点的 RNN 输出

 3. return_state - 默认 : False，是否返回 RNN state

```python
"""
设置call method:
1.inputs:(context, x)pair
    --context : is the context from the encoder's output.
    --x : is the target sequence input.
2.state: Optional, the previous state output from the decoder
3.return_state : [Default: False] - Set this to True to return the RNN state.
"""
@Decoder.add_method
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
    # print("x: ", x.shape)
    # (batch_size, 1(x_len), 256)
    
    #  generate the logits predcition
    logits = self.output_layer(x)
    
    if return_state:
        return logits, state
    else:
        return logits
```

其他方法生成方法

```python
@Decoder.add_method
"""
对Decoder进行初始化
Args: 
	context: Encoder 层输出
Returns:
	start_tokens: 包含start_token的Batch数据
	done: 标识是否生成完整句
	self.rnn.get_initial_state(embedded)[0]: 
"""
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    # rnn.get_initial_state:用Start初始化RNN
    initstate = self.rnn.get_initial_state(embedded)[0]
    print(initstate)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

@Decoder.add_method
"""
将tokens转化为文本
Args:
	tokens: token IDs
Returns:
	result: token 转化后的字符串
"""
def tokens_to_text(self, tokens):
    words = self.id_to_words(tokens)
    # 沿轴拼接字符串
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result
    
    
@Decoder.add_method
"""
根据已生成的文本生成下一个文本
Args:
	context: Encoder输出
	next_token: 生成的上一个token
	done: 标识是否已到句尾
	state: 已生成句子的 RNN 状态
Returns:
	next_token: 当前时间节点生成的token
	done: 标识是否已到句尾
	state: 当前 RNN 状态
"""
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
        logits = logits[:,-1,:]/temperature
        
        # logits:(batch_size, numsofclasses)
        # categorical:对logits的每行通过概率进行抽样
        next_token = tf.random.categorical(logits, num_samples=1)
        
    # If a sequence produces an 'end_token', set it 'done'
    done = done | (next_token == self.end_token)
    # 序列长度可能不一: 对结束了的句子用0-padding进行填补
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
    
    return next_token, done, state
```

测试序列生成代码

```python
"""
Generation loop-test
"""
decoder = Decoder(target_text_processor, UNITS)
next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []

for n in range(10):
    # Run one step
    next_token, done, state = decoder.get_next_token(
        ex_context, next_token, done, state, temperature=1.0)
    tokens.append(next_token)

# Stack all the tokens together.
tokens = tf.concat(tokens, axis=-1)

# convert the tokens back to a string
result = decoder.tokens_to_text(tokens)
print(result[:2].numpy())
print(tokens[:2].numpy())
```

```python
>> [b'accident theme beginners patients caused difficulty slim instructions babies silently'
 b'handbag dark slapped instantly their burned piano abuse virtue longest']
>> [[ 401 4835 4258 2561 1354 1465 4380 2356 3902 4007]
 [3788  542 3410 4116  236 1414  637 3360 3933 2723]]
```

#### 四、Model整合部分

##### 4.1 Translator类

​	将所有组件整合在Translator类中，以实现 End2End 模型，进行训练。

```python
"""
Combine all the components to a single model for training:
"""
class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
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
        
            #TODO(b/250038731): remove this
        try:
          # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
            del logits._keras_mask
        except AttributeError:
            pass
        
        return logits
```

测试Translator类：

```python
# 测试model
model = Translator(UNITS, context_text_processor, target_text_processor)

logits = model((ex_context_tok, ex_tar_in))

print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')
```

```python
>> Context tokens, shape: (batch, s, units) (64, 21)
>> Target tokens, shape: (batch, t) (64, 18)
>> logits, shape: (batch, t, target_vocabulary_size) (64, 18, 5000)
```

#### 六、模型训练

##### 6.1 Metric方法定义

- 损失函数

```python
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)
```

tf.keras.losses.SparseCategoricalCrossentropy()用于计算函数用于计算多分类问题的交叉熵，计算公式如下：

![img](https://img-blog.csdnimg.cn/a136f7fc093b444e81c793edd5297b25.png#pic_left)

其中，输入的y_true不应该是one-hot编码，而应该是整数列表；reduction变量可以控制是否对全局损失进行计算

```python
y_true = [0,2,1]
y_pred = [[0.6,0.1,0.1],[0.3,0.4,0.4],[0.3,0.3,0.4]]
# loss_1 输出每行的损失
loss_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# loss_2 会计算全局损失
loss_2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_1(y_true, y_pred))
print(loss_2(y_true, y_pred))
```

```python
>> tf.Tensor([0.7943768 1.0663774 1.1330688], shape=(3,), dtype=float32)
>> tf.Tensor(0.997941, shape=(), dtype=float32)
```

- 准确率函数

```python
def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)
```

由于模型的参数都是随机初始化的，按照公式得出理论的损失和准确率:

```python
model.compile(optimizer='adam',
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])

vocab_size = 1.0 * target_text_processor.vocabulary_size()

#  the initial values of the metrics should be:
{"expected_loss": tf.math.log(vocab_size).numpy(),
 "expected_acc": 1/vocab_size}
```

```python
>> {'expected_loss': 8.517193, 'expected_acc': 0.0002}
```

对 损失函数 和 准确率函数 进行测试，查看是否与理论值一致：

```python
model.evaluate(val_ds, steps=20, return_dict=True)
```

```python
>> {'loss': 8.545772552490234,
 'masked_acc': 0.00028224135166965425,
 'masked_loss': 8.545772552490234}
```

##### 6.2 模型训练

​	模型训练中使用了EarlyStopping的回调函数，在3次迭代未发生损失下降时，停止训练。

```python
history = model.fit(
    train_ds.repeat(), 
    epochs=100,
    steps_per_epoch = 100,
    validation_data=val_ds,
    validation_steps = 20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3)])
```

绘制损失值下降图：

```python
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
```

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230405113255084.png" alt="image-20230405113255084" style="zoom:67%;" />

绘制准确度上升图

```python
plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
```

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230405113406886.png" alt="image-20230405113406886" style="zoom:67%;" />

#### 七、END2END实现

​	模型已经训练好了，为 Translator 类定义一个translate()方法，可实现从Spanish到English的端到端翻译

```python
"""
The model has been trained, now to implement the "text=>text" translation
"""
@Translator.add_method
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
                       
    # 在列维度 concat
    tokens = tf.concat(tokens, axis=-1)
    # 在(batch, query_seq, key_seq)中的query_seq维度concat
    self.last_attention_weights = tf.concat(attention_weights, axis=1)
    
    result = self.decoder.tokens_to_text(tokens)
    return result
```

测试 translate()方法：

```python
result = model.translate(['¿Todavía está en casa?']) # Are you still home
result[0].numpy().decode()
```

```
>> 'is he still at home ? '
```

- 官方还给出了Attention 权重图的绘制函数plot_attention()，可以直观地看到不同query和key值之间的权重：

```python
@Translator.add_method
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
```

```python
model.plot_attention('¿Todavía está en casa?') # Are you still home
```

<img src="C:\Users\Winter Wei\AppData\Roaming\Typora\typora-user-images\image-20230405152223300.png" alt="image-20230405152223300" style="zoom:80%;" />

#### 八、模型存储与读取

##### 8.1 Export类与tf.function

​	tensorflow2.12提供了tf.function可以实现对模型方法进行打包：

```python
class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)
```

将我们训练好的 model 对Export类进行初始化，并可通过export调用translate方法进行翻译。

```python
export = Export(model)   
inputs=[
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.' # "His room is a mess"
    ]
result = export.translate(tf.constant(inputs))
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
```

```python
>> it is very cold here .                                            
>> this is my life .                                             
>> her room is a disaster . 
```

##### 8.2 function方法存储

​	在添加了@tf.function注解后，translate()便可被追踪，可使用saved_model.save进行存储：

```python
tf.saved_model.save(export, 'translator',
                    signatures={'serving_default': export.translate})
```

##### 8.3 模型读取

同样，我们也可以使用save_model.load()方法读取模型。不过，由于我们只保存了原模型的translate方法，因此我们读取后的模型，也只能使用translate方法。

```python
model = tf.saved_model.load('translator')
result = model.translate(tf.constant(inputs))
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
```

```python
>> it is very cold here .                                            
>> this is my life .                                             
>> her room is a disaster . 
```

#### 九、总结

​	至此，基于Attention的机器翻译模型已成功实现。在此项目中，我们使用Tensorflow提供的翻译数据，使用Attention机制实现了Spanish - English的整句翻译。

​	模型分为Encoder和Decoder两部分。在Encoder层中，实现了对输入文本context语义信息的获取；在Decoder层中，通过Attention机制实现了对Encoder语义信息的使用，使用RNN顺序生成文本。

​	本项目只是简单地使用attention进行了单语言翻译的实现，未来还可以根据不同数据集进行不同语种翻译的实现；同时，本项目的模型Encoder和Decoder均使用了RNN来获取上下文信息，这同样会导致大量信息的遗失。一个较好的解决办法是使用Transformer 层来代替 RNN；同时，在Tokenization层上同样可以进行优化，例如使用BertTokenizer/embedding来实现词语的token化和embedding。

#### 十、附录

