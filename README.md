# AttnTranslator
Translation of Spanish to English with the attention and RNN layers
### Introduction
This project is a translation model from Spanish to English, built with tensorflow(v2.12.0). It is from [Neural machine translation with attention](https://tensorflow.google.cn/text/tutorials/nmt_with_attention)
### Structure
DataLoader.py: Load online data of Spanish-English pairs from tensorflow.
Utils.py: Some util functions.
train.py: Train the model and save it.
### Run the model
You can train the model by the following codes
```
python train.py --plotting True --saving True --batch_size 64 --epochs 100 --UNITS 256
```
You can run use the model to conduct translation jobs through ```python predict.py```
  
The detailed descriptions are in the 
