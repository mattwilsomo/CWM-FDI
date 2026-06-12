import modelling


class Model:
    def __init__(self, modeltype, filepath=None):
        self.modeltype = modeltype
        self.filepath = filepath

        self.model = None
        self.train_time = None
        self.inference_time = None

        self.train_energy = None
        self.inference_energy = None
        self.carbon_footprint = None

        self.ap_score = None
        self.X_test = None
        self.y_test = None

    def train(self):
        match self.modeltype.lower():
            case "decision tree" | "dt":
                result = modelling.DecisionTree(self.filepath)
            case "logistic regression" | "lr":
                result = modelling.LogRegression(self.filepath)
            case "random forest" | "rf":
                result = modelling.RandomForest(self.filepath)
            case "svm":
                result = modelling.SupportVectorMachine(self.filepath)
            case "mlp":
                result = modelling.MLP(self.filepath)
            case _:
                print(
                    "Not a valid model, the options are:\n"
                    "-decision tree/dt\n"
                    "-logistic regression/lr\n"
                    "-random forest/rf\n"
                    "-svm\n"
                    "-mlp"
                )
                return

        self.model, self.train_time, self.X_test, self.y_test = result

    def evaluate(self):
        self.ap_score, self.inference_time = modelling.evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
        )


if __name__ == "__main__":
    m = Model("lr", "../data/creditcard.csv")
    m.train()

    print("Model:", m.model)
    print("Train time:", m.train_time)
    print("X test shape:", m.X_test.shape)
    print("Y test shape:", m.y_test.shape)

    m.evaluate()

    print("AP score:", m.ap_score)
    print("Inference time:", m.inference_time)
