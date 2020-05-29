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


4. predict
- produce submission file

- with Bert fine tuning model
	  
      python3 cls.py --test_data_path ./data/test.csv --mode test --load_model ./model/path/to/your/model

- with XLNet fine tuning model
 
	  python3 cls.py --test_data_path ./data/test.csv --mode test --xlnet --load_model ./model/path/to/your/model