1. add data (train.csv, test.csv, submission.csv) in data folder

2. data preprocess

       python3 makedata.py --path data/train.csv

   or xlnet data preprocess

       python3 makedata.py --path data/train.csv --xlnet

3. training
- train with bert:

      python3 cls.py --mode train

- or train with xlnet

      python3 cls.py --mode train --xlnet



