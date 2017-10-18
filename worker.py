# @a hui

import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm

job_mapping = {
	 'admin.': 1,
	 'blue-collar': 2,
	 'entrepreneur': 3,
	 'housemaid': 4,
	 'management': 5,
	 'retired': 6,
	 'self-employed': 7,
	 'services': 8,
	 'student': 9,
	 'technician': 10,
	 'unemployed': 11,
	 'unknown': 0,
}

marital_mapping = {
	'divorced': 1,
	'married': 2,
	'single': 3,
	'unknown': 0,
}

education_mapping = {
	'basic.4y': 1,
	'basic.6y': 2,
	'basic.9y': 3,
	'high.school': 4,
	'illiterate': 5,
	'professional.course': 6,
	'university.degree': 7,
	'unknown': 0,
}

credit_mapping = {
	'no': 1,
	'yes': 2,
	'unknown':0,
}

house_load_mapping = {
	'no': 1,
	'yes': 2,
	'unknown':0,
}

personal_load_mapping = {
	'no': 1,
	'yes': 2,
	'unknown':0,
}

contact_mapping = {
	'cellular': 1,
	'telephone': 2,
}

month_mapping = {
	'jan': 1,
	'feb': 2,
	'mar': 3,
	'apr': 4,
	'may': 5,
	'jun': 6,
	'jul': 7,
	'aug': 8,
	'sep': 9,
	'oct': 10,
	'nov': 11,
	'dec': 12,
}

day_of_week_mapping = {
	'mon': 1,
	'tue': 2,
	'wed': 3,
	'thu': 4,
	'fri': 5,
}

poutcome_mapping = {
	'failure': 1,
	'nonexistent': 0,
	'success': 2
}

label_mapping = {
	'yes': 1,
	'no': 0,
}


def convert_feature(i, feature):
	# Can apply more advanced feature conversion.
	val = 0
	if i == 2:
		# Job
		val = job_mapping[feature] 

	elif i == 3:
		# Marital
		val = marital_mapping[feature]

	elif i == 4:
		# Education
		val = education_mapping[feature]

	elif i == 5:
		# Credit	
		val = credit_mapping[feature]

	elif i == 6:
		val = house_load_mapping[feature]

	elif i == 7:
		val = personal_load_mapping[feature]

	elif i == 8:
		val = contact_mapping[feature]

	elif i == 9:
		val = month_mapping[feature]

	elif i == 10:
		val = day_of_week_mapping[feature]

	elif i == 15:
		val = poutcome_mapping[feature]

	else:
		val = float(feature)

	return val


def read_file(filename, with_label = True):
	data_list = []
	label_list = []
	
	with open(filename) as file:
		reader = csv.reader(file, delimiter=',')

		for row in reader:
			data = []

			for i in range(len(row)):
				if with_label and i == len(row)-1:
					# Skip the label.
					continue

				val = convert_feature(i, row[i])
				data.append(val)

			data_list.append(data)
			
			if with_label:
				label = label_mapping[row[len(row)-1]]
				label_list.append(label)

	features = np.asarray(data_list)
	labels = np.asarray(label_list)
	return features, labels


def write_file(filename, predictions):
	with open(filename, 'w') as csvfile:
		fieldnames = ['id', 'prediction']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
		for i in range(len(predictions)):
			writer.writerow({'id': i, 'prediction': predictions[i]})


def accurarcy(y, y_predicted):
	count = 0.0
	for i in range(len(y_predicted)):
		if abs(y_predicted[i] - y[i]) < 0.001:
			count = count + 1

	accurarcy = count / y.shape[0]
	return accurarcy


# Config values
features_to_remove = [0, 11,]
fold = 3
##########################

train_features, train_labels = read_file('train.csv')
number = train_features.shape[0]
split = number * fold / (fold + 1)
validation_features, validation_labels = train_features[split:, :], train_labels[split:]
train_features, train_labels = train_features[:split, :], train_labels[:split]
test_features, _ = read_file('test.csv', False)

print train_features.shape, test_features.shape

# Feature selection
for feature_to_remove in features_to_remove:
	train_features = np.delete(train_features, feature_to_remove, 1)
	validation_features = np.delete(validation_features, feature_to_remove, 1)
	test_features = np.delete(test_features, feature_to_remove, 1)

print ('After feature selection ', train_features.shape, test_features.shape)

# Model training
# Can try with different models
clf = linear_model.SGDClassifier()
# clf = svm.SVC()
###############################

clf.fit(train_features, train_labels)

predicted_training_label = clf.predict(train_features)
print ('Accurarcy of training data set ', accurarcy(train_labels, predicted_training_label))

predicted_validation_label = clf.predict(validation_features)
print ('Accurarcy of training data set ', accurarcy(validation_labels, predicted_validation_label))

predicted_test_label = clf.predict(test_features)
print ('Write to csv')
write_file('Submission.csv', predicted_test_label)
