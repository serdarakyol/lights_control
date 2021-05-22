# Light of Controls
You can create your own dataset, train and predict with this repository.
# USUAGE
```
$ virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
* After these steps, if you don't have any dataset, you can create you worn dataset while using prepareDataset.py
* Then you need to convert you wav data to json format for you can implement CNN to your own dataset
* After you can just go to train.py and start to train your model. Remember if you have different input shape than me, you will need little preprocessing for your dataset.
* The last step is only you have to run predictWord.py file for you can make predictions.
