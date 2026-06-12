import modelling 

class Model():
	def __init__(self, modeltype, filepath=None):
		self.inference_time = None 
		self.modeltype = modeltype
 		self.filepath = filepath
		self.model = None
		self.inference_time = None
		self.inference_energy = None
		self.accuracy = None
		self.train_time = None
		self.train_energy = None 
		self.carbon_footprint = None
		self.X_test = None
		self.y_test = None

	def train(self):

		match self.modeltype.lower():
			case "decision tree" | "dt":
				result = modelling.DecisionTree()
			case "logistic regression" | "lr":
				result = modelling.LogRegression()
			case "random forest" | "rf":
				result = modelling.RandomForest()
			case "svm":
				result = modellling.SupportVectorMachine()
			case "mlp": 
				result = modelling.MLP()
			case _: 
				print("Not a valid model, the options are: \n-decision tree/dt\n-logistic regression/lr\n-random forest/rf\n-svm\n-mlp")
				return
			self.model, self.train_time,self.X_test, self.y_test = result


 
