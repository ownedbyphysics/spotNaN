# spotNaN
Fun project to spot all kinds of null values of a dataset

Sometime ago I had to process large amounts of datasets full of nulls which were in any kind of string formats (na-n-..-/n/-NaT-]na etc) since it was text-revies datasets.
Had an idea to train a neural network just for fun to read this datasets and predict all cases that resemble null values. 
The class via -predict method- returns any kind of testing dataset having predicted Nulls and having converted them to np.null type so you can get rid of them easily.
