from data_loading import load_data
from sklearn.ensemble import LogisticRegression
from sklearn import tree
from sklearn import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
import time 


def split(filepath: str) -> tuple:
	df = load_data(filepath)
	
	# Splits the dataframe into the training vector and the target vector 
	X = df.drop("Class", axis =1) # training vector
	y = df["Class"] #target vector 

	#Will split up the data into a train section and a train section
	#Tests the model on 20% of the data, random_state = 42 to get the same results each time (keep pseudorandom state), statify for use with small number of targets  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)  

	return (X_train, X_test, y_train, y_test)

def DecisionTree(filepath: str = "../data/creditcard.csv"):

	# Will train a logistic regression model on the data 

	X_train, X_test, y_train, y_test = split(filepath)

	start_train = time.time()

	#trains the model on the data, cap the tree depth to avoid overfitting
	model = tree.DecisionTreeClassifier(class_weight = "balanced", random_state = 42, max_depth = 10).fit(X_train, y_train)
	
	end_train = time.time()
	train_time = end_train - start_train 


	return (model, train_time, X_test, y_test)

def LogRegression(filepath: str = "../data/creditcard.csv"):

        # Will train a logistic regression model on the data 

        X_train, X_test, y_train, y_test = split(filepath)

        start_train = time.time()

        # creates the pipeline object, this combines the preprocessing and the model 
        #The standard scaler will remove the mean and scale with a unit variance
        #increased iteration count to allow convergence 
        #balanced uses the values of y to adjust the weights inversely proportional to frequ>
        pipe= make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter = 1000,class_weight="balanced" ),
        )


        logistic = pipe.fit(X_train, y_train)

        end_train = time.time()
        train_time = end_train - start_train 

def RandomForest(filepath: str = "../data/creditcard.csv"):

        # Will train a logistic regression model on the data 

        X_train, X_test, y_train, y_test = split(filepath)

        start_train = time.time()

        #trains the model on the data, cap the tree depth to avoid overfitting
	model = RandomForestClassifier(class_weight = "balanced", random_state = 42, max_depth = 10).fit(X_train, y_train)


        end_train = time.time()
        train_time = end_train - start_train 


        return (model, train_time, X_test, y_test)

def SupportVectorMachine(filepath: str = "../data/creditcard.csv"):

        # Will train a SVM model on the data 

        X_train, X_test, y_train, y_test = split(filepath)

        start_train = time.time()

	#Creates pipeline
	#caps the iterations to avoid overfitting
	pipe = make_pipeline(
		StandardScaler(),
		svm.LinearSVC(class_weight = "balanced", random_state = 42, max_iter = 5000)
	)

        #trains the model on the data
        model = pipe.fit(X_train, y_train)


        end_train = time.time()
        train_time = end_train - start_train 


        return (model, train_time, X_test, y_test)
