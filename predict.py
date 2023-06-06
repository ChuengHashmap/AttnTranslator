from Models import Export
import tensorflow as tf


def predict():
    model = tf.saved_model.load('translator')
    inputs = [
        'Hace mucho frio aqui.',  # "It's really cold here."
        'Esta es mi vida.',  # "This is my life."
        'Su cuarto es un desastre.'  # "His room is a mess"
    ]
    results = model.translate(tf.constant(inputs))
    tf.print(inputs[0], ' -> ', results[0])
    tf.print(inputs[1], ' -> ', results[1])
    tf.print(inputs[2], ' -> ', results[2])

if __name__=='__main__':
    predict()
