import numpy, pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


if __name__=="__main__":
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	# load dataset
	dataframe = pandas.read_csv('iris.csv', header=None)
	dataset = dataframe.values
	X = dataset[:,0:4].astype(float)
	Y = dataset[:,4]

	# encode output variables
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)
	# create model
	estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
	# evaluate model wit k-fold cross validation
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X, dummy_y, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))