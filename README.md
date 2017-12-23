# Cross_Domain_Hedge_Detection
We implement the cross domain Chinese hedge detection using Keras.
## DataSet
This dataset contains four domains: wiki, biomedical abstract, discuss and result. 

For example, we use the abstract to train and test for the wiki, this is called wiki_by_abstract (test_by_train). 

We use five fold test to evaluate our model.

### word embedding
We train our word embedding on a small corpus downloaded from the Medline, we will make our word embedding public as soon as possible. <br>
We would like to make our data public as soon as possible.
## Use
* python 2.7
* Keras 2.0.1
* Tensorflow 1.0.1
* nltk 3.2.2
* tqdm
## Run
To creature the data and features
```bash
python hedge_process.py
```
To process the data into the matrix and use for learning
```bash
python process_data.py
```
To run the BiLSTM model
```bash
python main.py
```
