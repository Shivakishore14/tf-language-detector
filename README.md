# tf-language-detector

An CNN based text classification used for predicting the language used in a source file.

# Dependencies

install dependencies via pip

```
pip install -r requirements.txt
```

# Configuration

configure the config.py as needed for the dataset

the default config.py configuration expects the dataset to be in format of

```
project-root
    |-dataset
         |- language1
         |     |- samplefile1.ext1
         |     |- ...
         |     |- samplefile100.ext1
         |- language1
               |- samplefile1.ext2
               |- ...
               |- samplefile100.ext2

```

# Dataset

its easy to generate dataset with this simple bash script

```
cp $(find <DIR> -type f -name "*.<EXTENSION>") ./dataset/<LANG1>/
```

NOTE: <br>
replace `<DIR>` with a directory witch has a lot of programs / scripts for the particular `<EXTENSION>` <br>
replace `<EXTENSION>` with a extention of your language

Example : `cp $(find /home/users/sk/github/ -type f -name "*.go") ./dataset/go/`

# Usage

for training use `python train.py` <br>
for prediction use `python predict.py` or flask endpoint with `python predict-server.py` and post file text to /predict<br>
