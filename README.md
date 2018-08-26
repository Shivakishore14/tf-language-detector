# tf-language-detector


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

use train.py for training and predict.py for testing

export graph with freeze.py after training for prediction.


# Dataset

its easy to generate dataset with this simple bash script

```
cp $(find <DIR> -type f -name "*.<EXTENSION>") ./dataset/<LANG1>/
```

NOTE: <br>
replace `<DIR>` with a directory witch has a lot of programs / scripts for the particular `<EXTENSION>` <br>
replace `<EXTENSION>` with a extension your extention of your language

Example : `cp $(find /home/users/sk/github/ -type f -name "*.go") ./dataset/go/`

