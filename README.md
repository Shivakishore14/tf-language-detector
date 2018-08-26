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

export graph with freeze.py after training.
