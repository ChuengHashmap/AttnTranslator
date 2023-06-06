import tensorflow as tf
import pathlib
import numpy as np
from utils import tf_lower_and_split_punct


def load_data():
    # 加载keras中的数据
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip',
                                          origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                                          extract=True)
    path_to_file = pathlib.Path(path_to_zip).parent / 'spa-eng/spa.txt'

    text = path_to_file.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    context_raw = np.array([context for _, context in pairs])  # 原句
    target_raw = np.array([target for target, _ in pairs])  # 翻译句子
    return context_raw, target_raw


def create_dataset(batch_size, train_split=0.8):
    BATCH_SIZE = batch_size
    context_raw, target_raw = load_data()
    BUFFER_SIZE = len(context_raw)

    is_train = np.random.uniform(size=(len(target_raw))) < train_split
    train_raw = (tf.data.Dataset
                 .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 )

    val_raw = (tf.data.Dataset
               .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
               .shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE)
               )

    max_vocab_size = 5000
    # Spainish TextVectorization
    context_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True
    )
    context_text_processor.adapt(train_raw.map(lambda context, target: context))

    # English TextVectorization
    target_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True
    )
    target_text_processor.adapt(train_raw.map(lambda context, target: target))

    """
    将 context, target转化为模型所需要得输入：
    inputs: (context, target_in)
    outputs: target_out
    其中target_in 与 target_out刚好错位，target_out中的元素对应着target_in的下一个元素
    """

    def process_text(context, target):
        context = context_text_processor(context).to_tensor()
        target = target_text_processor(target)
        target_in = target[:, :-1].to_tensor()
        target_out = target[:, 1:].to_tensor()
        return (context, target_in), target_out

    # 将train_ds, val_ds全部转化为了id
    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    return train_ds, val_ds, context_text_processor, target_text_processor
