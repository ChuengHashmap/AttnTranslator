import tensorflow as tf
from DataLoader import *
from Models import *
from utils import masked_acc, masked_loss
import argparse

parser = argparse.ArgumentParser(description='Tensorflow Translator')
parser.add_argument('--plotting', type=bool, default=False, help='')
parser.add_argument('--saving', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--UNITS', type=int, default=256, help='')
parser.add_argument('--epochs', type=int, default=100, help='')
args = parser.parse_args()


def train(ploting=False, saving=True):
    train_ds, val_ds, context_text_processor, target_text_processor = create_dataset(batch_size=args.batch_size)
    model = Translator(args.UNITS, context_text_processor, target_text_processor)
    model.compile(optimizer='adam',
                  loss=masked_loss,
                  metrics=[masked_acc, masked_loss]
                  )

    history = model.fit(
        train_ds.repeat(),
        epochs=args.epochs,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3)])

    if ploting == True:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch #')
        plt.ylabel('CE/token')
        plt.legend()

        plt.plot(history.history['masked_acc'], label='accuracy')
        plt.plot(history.history['val_masked_acc'], label='val_accuracy')
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch #')
        plt.ylabel('CE/token')
        plt.legend()

    if saving == True:
        export = Export(model)
        tf.saved_model.save(export, 'translator', signatures={'serving_default': export.translate})


if __name__ == '__main__':
    train()
