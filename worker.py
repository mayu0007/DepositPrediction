# @a hui

import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn import feature_selection
import heapq


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


# Config values 11 19 20 1 16 18
features_to_remove = [0] #[0, 6, 7, 10]  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, ]
fold = 9
##########################

train_features, train_labels = read_file('train.csv')
# train_features = normalize(train_features)
number = train_features.shape[0]
split = number * fold / (fold + 1)
validation_features, validation_labels = train_features[split:, :], train_labels[split:]
train_features, train_labels = train_features[:split, :], train_labels[:split]
test_features, _ = read_file('test.csv', False)

# Feature selection
train_features = np.delete(train_features, features_to_remove, 1)
validation_features = np.delete(validation_features, features_to_remove, 1)
test_features = np.delete(test_features, features_to_remove, 1)

print ('After feature selection ', train_features.shape, test_features.shape)

# Over sample train data
'''
selectKBest = SelectKBest(feature_selection.mutual_info_classif)
selectKBest.fit(train_features, train_labels)
scores = selectKBest.scores_
print sum(scores)
print scores
print heapq.nlargest(20, range(len(scores)), scores.__getitem__)
'''

'''
# kind = ['regular', 'borderline1', 'borderline2', 'svm']
kind = ['svm']
sm = [SMOTE(kind=k) for k in kind]
for method in sm:
	X_res, y_res = method.fit_sample(train_features, train_labels)
	# print 'X_res', X_res
	train_features = np.append(train_features, X_res, axis=0)
	train_labels = np.append(train_labels, y_res, axis=0)

print ('After oversampling ', train_features.shape)
'''
'''
# Apply Nearmiss
version = [3]
nm = [NearMiss(version=v, return_indices=True) for v in version]

X_resampled = []
y_resampled = []
for method in nm:
    X_res, y_res, _ = method.fit_sample(train_features, train_labels)
    X_resampled.extend(X_res)
    y_resampled.extend(y_res)

train_features = np.asarray(X_resampled)
train_labels = np.asarray(y_resampled)
'''

# Model training
# Can try with different models

weights = {0:1, 1:5}
clf = linear_model.LogisticRegression(class_weight='balanced', max_iter=5000)

# clf = svm.SVC()
###############################

clf.fit(train_features, train_labels)

predicted_training_label = clf.predict(train_features)
print ('Accurarcy of training data set ', matthews_corrcoef(train_labels, predicted_training_label))

predicted_validation_label = clf.predict(validation_features)
print ('Accurarcy of training data set ', matthews_corrcoef(validation_labels, predicted_validation_label))

predicted_test_label = clf.predict(test_features)
print predicted_test_label
print ('Write to csv')
write_file('Submission.csv', predicted_test_label)
