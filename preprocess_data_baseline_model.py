import sys
import csv
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

csv.field_size_limit(sys.maxsize)
model_d = {"DecisionTree":1, "RandomForest":2, "KNeighbors":3, "GaussianNB":4, "LR":5, "LinearSVC":6}
model_number = model_d[sys.argv[1]]
# numeric data column standardization in scale
def scale(x):
	x = np.asarray(x, dtype=float)
	min_max_scaler = preprocessing.MinMaxScaler()
	x_minmax = min_max_scaler.fit_transform(x)
	x_minmax = np.array([x_minmax])
	return x_minmax.T

# categorical data column standardization in scale
def enc(x):
	enc = preprocessing.OneHotEncoder()
	x_minmax = enc.fit_transform(x).toarray()
	return x_minmax
# Map data time to month
def map_date_month(date):
	date = date.strip()
	if len(date) > 15:
		date_object = datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
	else:
		date_object = datetime.strptime(date,"%Y-%m-%d")
	month = str(date_object.month)
	return month

def date_duration(x,y):
	x = x.strip()
	y = y.strip()
	date_object_x = datetime.strptime(x,"%Y-%m-%d")
	date_object_y = datetime.strptime(y,"%Y-%m-%d")
	timedelta = date_object_y - date_object_x
	days = float(timedelta.days)
	return days

def orig_destination_distance_tranform(x):
	if len(x)<1:
		return 0.0
	else:
		return float(x)

# Load data
X = []
Y = []
train_is_booking = "../Kaggle-Data/train_is_booking.csv"
f = open(train_is_booking,'rb')
r = csv.DictReader(f)

#limit_num = 0
user_dict = {}
current_row = 0


for row in r:
	#if limit_num > 100000:
		#break

	if row['user_id'] not in user_dict:
		user_dict[row['user_id']] = [current_row]
	else:
		user_dict[row['user_id']].append(current_row)

	distance = orig_destination_distance_tranform(row['orig_destination_distance'])
	row_X = [row['user_id'], map_date_month(row['date_time']), row['site_name'], 
				row['posa_continent'], row['user_location_country'], #row['user_location_region'], 
				distance, row['is_mobile'], row['is_package'],
				row['channel'], map_date_month(row['srch_ci']), date_duration(row['srch_ci'], row['srch_co']), 
				row['srch_adults_cnt'], row['srch_children_cnt'], row['srch_rm_cnt'],
				row['srch_destination_type_id'], row['hotel_continent'], row['hotel_country']
				]
	row_Y = row['hotel_cluster']
	#limit_num += 1
	X.append(row_X)
	Y.append(row_Y)
	current_row  += 1

X = np.asarray(X, dtype='float64')
Y = np.asarray(Y)

distance = scale(X[:,5])
other_features = enc(X[:,[1,2,3,4,6,7,8,11,12,13,14,15,16]])
X = np.hstack((distance, other_features))
print "X and Y shape", X.shape, Y.shape
#X = csr_matrix(X)

name_list = user_dict.keys()
total_set_length = len(name_list)
kf = KFold(total_set_length, n_folds=5)
avg_score = []
for train_index, test_index in kf:
	temp_index = []
	for i in train_index:
		temp_index += user_dict[name_list[i]]
	train_index = np.array(temp_index)
	temp_index = []
	for j in test_index:
		temp_index += user_dict[name_list[j]]
	test_index = np.array(temp_index)
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
	if model_number == 1:
		clf = DecisionTreeClassifier(random_state=0)
	elif model_number == 2:
		clf = RandomForestClassifier(n_estimators=10)
	elif model_number == 3:
		clf = KNeighborsClassifier(n_neighbors=100, weights='distance')
	elif model_number == 4:
		clf = GaussianNB()
	elif model_number == 5:
		clf = linear_model.LogisticRegression(penalty='l2',solver='lbfgs', multi_class='multinomial')
	elif model_number == 6:
		clf = OneVsRestClassifier(LinearSVC(random_state=0))
	clf.fit(X_train, Y_train)
	y_true, y_pred = Y_test, clf.predict(X_test)
	score = clf.score(X_test, Y_test)
	avg_score.append(score)
	print score
print "avg score is:",np.mean(avg_score)
#scale:X[:,5]
#enc: X[:,[1,2,3,4,6,7,8,11,12,13,14,15,16]]

# date_time map into month
# orig_destination_distance ->  preprocessing.MinMaxScaler()

# srch_ci -> map to month
# srch_ci and srch_co -> duration

# srch_adults_cnt, srch_children_cnt, srch_rm_cnt -> preprocessing.MinMaxScaler()

# if is_booking == 1, cnt usually equals 1. We convert any value large than 1 to 1
# if is_booking == 0, the higher the cnt the more a user is interested in a given hotel.


# Based on 100k data. 
# Decision Tree:0.0456617466498
# Random Forest:0.0437495357021
# GaussianNB: 0.010811611752
# Onevsrest LinearSVC: 0.0851954496301
# LR:0.0851954496301
