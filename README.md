# donut_demo

## donut investigation

   [詳細な説明を見る](docs/donutmemo.md)
   
   [学習データセット作成](docs/makedataset.md)

## references
#### sample codes
    https://huggingface.co/docs/transformers/en/model_doc/donut

#### train
    https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut


## MAKE ENVIOROMENT PYTHON VENV

    python3 -m venv venv


##### install packages

    source venv/bin/activate

###### datasets
    pip install datasets

###### deeplearning
    pip install transformers
    pip install torch 
###### data fast tokenizer
    pip insall sentencepiece

###### pacage loader
    pip install protobuf
###### ui
    pip install flet 
    pip install opencv-python 
    pip install pillow

## docParsing samples
#### run sample
    cd donut_demo
    python3 DocumentParsing_sample.py


#### run webcam sample
    python3 ./docParsing/docParsingMain.py
