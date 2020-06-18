1. add data (train.csv, test.csv, submission.csv) in data folder

2. bert data preprocess

       python3 makedata.py --path data/train.csv --bert

   or roberta data preprocess

       python3 makedata.py --path data/train.csv --roberta

3. training
- train with bert:

      python3 cls.py --mode train --bert

- or train with roberta

      python3 cls.py --mode train --roberta


4. predict
- produce submission file

- with Bert fine tuning model
	  
      python3 cls.py --mode test --bert --load_model ./model/task1-bert.pth --test_data_path ./data/test.csv --output_path ./task1_method1.csv

- with XLNet fine tuning model
 
	  python3 cls.py --mode test --roberta --load_model ./model/task1-roberta.pth --test_data_path ./data/test.csv --output_path ./task1_method2.csv
