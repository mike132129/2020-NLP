# Causal Sentence Prediction

1. For training:

- Firstly, data preprocessing:

    `$ python3 makedata.py --train --path ./data/train.csv --xlnet(optional)`

	Then the preprocessing file will be store in data/ .

- Go training:

	`$ python3 causal.py --mode train > training.log`

- You are able to check the training loss and validation result in the training log.

2. For produce prediction:

- Data preprocessing

	`$ python3 makedata.py --path ./data/test.csv --xlnet(optional)`

- Produce prediction file

	`$ python3 causal.py --mode test --test_data_path ./data/test.csv`

3.  Evaluate the result:

	`$ python3 task2_evaluate.py from-file --ref_file data/split_dev.csv train_predict.csv`