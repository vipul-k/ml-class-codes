import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import make_scorer
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def qwk(y1, y2):
    
    #quadratic weighted kappa
    
    levels = np.unique(y1)  
    k = len(levels)
    obs = len(y1)
    print()
    O = np.empty([k,k])
    E = np.empty([k,k])
    W = np.empty([k,k])
    for i in range(k):
        for j in range(k):
            O[i,j] = ((y1 == levels[i]) & (y2 == levels[j])).sum()/obs
            E[i,j] = ((y1 == levels[i]).sum()/obs)*((y2 == levels[j]).sum()/obs)
            W[i,j] = np.power(levels[i]-levels[j], 2)
            
    kappa = 1 - ((W*O).sum()/(W*E).sum())
    return kappa  

class Model():
    def __init__(self, model = "Decision Tree", data_path = "train.csv", scale = False):
        self.model = model
        self.scale = scale
        print("Scale: ", self.scale)
        self.data_path = data_path
        
        categorical_vars = "Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41"
        
        continuous_vars = "Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5"
        
        discrete_vars = "Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32"
        
        self.categorical_list = categorical_vars.split(", ")
        self.continuous_list = continuous_vars.split(", ")
        self.discrete_list = discrete_vars.split(", ")
        self.dummy_list = ["Medical_Keyword_"+str(i) for i in range(1,49)]
        self.ohe_vars = []
        self.param_name = ""
        self.param_range = []
        
    def loadAndSample(self):
        '''
        The function loads the training data in a dataframe and takes a 
        10% sample to work with.

        Returns
        -------
        train: pandas dataframe

        '''
        train_data = pd.read_csv(self.data_path)
        train = train_data.sample(frac = 0.1, random_state = 44, 
                                  ignore_index = True)
        print("Number of rows in data: "+ str(train.shape[0]))
        
        return train
    
    def impute(self, train):
        '''
        The function imputes the missing value in the data using following 
        strategy:
            categorical variables: most frequent value
            discrete variable: most frequent value
            dummy variable: most frequent value
            continuous variable: mean value
            
        Categorical, discrete and dummy variable are imputed based on most 
        frequent value or mode value of data as most if the data is missing 
        because of error in recording data, mode value has the highest
        probability of occuring in the missing rows. Continuous variable are 
        imputed with mean value as that keeps the mean and standard deviation 
        of the data unchanged.

        Returns
        -------
        train_imp: imputed dataset

        '''
        
        
        imp = SimpleImputer(strategy="most_frequent")
        df_discrete = pd.DataFrame(columns = self.discrete_list, data = imp.fit_transform(train[self.discrete_list]))
        df_categorical = pd.DataFrame(data = imp.fit_transform(train[self.categorical_list]), columns = self.categorical_list)
        df_dummy = pd.DataFrame(data = imp.fit_transform(train[self.dummy_list]), columns = self.dummy_list)
        
        imp = SimpleImputer(strategy="mean")
        df_continuous = pd.DataFrame(data = imp.fit_transform(train[self.continuous_list]), columns = self.continuous_list)
        
        train_imp = pd.concat([df_discrete, df_categorical, df_continuous, df_dummy, train[["Id", "Response"]]], axis = 1)
        
        print("Shape of imputed data: " + str(train_imp.shape))
        
        return train_imp
    
    def one_hot_encode(self, train_imp):
        '''
        Create dummy indicator variables for categories of categorical variables

        Parameters
        ----------
        train_imp : pandas dataframe

        Returns
        -------
        train_final: pandas dataframe

        '''
        train_ohe = pd.get_dummies(train_imp[self.categorical_list], drop_first = True)
        self.ohe_vars = list(train_ohe.columns)
        
        train_final = pd.concat([train_imp, train_ohe], axis = 1)
        
        return train_final
    
    def scale_data(self, train_final):
        '''
        minmax scale the data

        Parameters
        ----------
        train_final : pandas dataframe

        Returns
        -------
        df : pandas dataframe

        '''
        x = train_final[self.discrete_list + self.continuous_list + self.ohe_vars + self.dummy_list]
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns = self.discrete_list + self.continuous_list + self.ohe_vars + self.dummy_list)
        
        df.loc[:,"Id"] = train_final["Id"]
        df.loc[:,"Response"] = train_final["Response"]
        print("Features Scaled")
        
        return df
    
    def curve_plots(self):
        train = self.loadAndSample()
        train_imp = self.impute(train)
        train_final = self.one_hot_encode(train_imp)
        if self.scale:
            train_final = self.scale_data(train_final)
        if self.model == "Decision Tree":
            param_name = "max_depth"
            params = [i for i in range(2,21,2)]
            clf = tree.DecisionTreeClassifier()
        if self.model == "Logistic Regression":
            param_name = "C"
            params = [i for i in range(1, 1001, 100)]
            clf = LogisticRegression(max_iter=3000, solver='lbfgs', n_jobs = -1)
            
        train_features = self.discrete_list + self.continuous_list + self.ohe_vars + self.dummy_list
            
        train_scores, valid_scores = validation_curve(clf ,X = train_final[train_features],
                                            y = train_final["Response"], param_name = param_name, 
                                            param_range = params, cv = 5, scoring = make_scorer(cohen_kappa_score, weights="quadratic"))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        
        
        plt.title("Validation Curve with Decision Tree")

        plt.plot(params, train_scores_mean, label = "Training accuracy")
        plt.plot(params, valid_scores_mean, label = "Validation accuracy")
        plt.legend()
        plt.xlabel(param_name)
        plt.ylabel("Score")
        #plt.savefig('validation.pdf')
        plt.show()
        
        best_param_index = int((np.where(valid_scores_mean == valid_scores_mean.max()))[0])
        best_param = params[best_param_index]
        
        if self.model == "Decision Tree":
            clf = tree.DecisionTreeClassifier(max_depth = best_param)
        if self.model == "Logistic Regression":
            clf = LogisticRegression(max_iter=3000, solver='lbfgs', n_jobs=-1, C = best_param)
            
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        train_sizes, train_scores, valid_scores = learning_curve(clf, X = train_final[train_features],
                                            y = train_final["Response"], train_sizes=train_sizes,
                                                         cv=5, scoring = make_scorer(cohen_kappa_score, weights="quadratic"))
        
        
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        
        plt.title("Learning Curve")
        
        plt.plot(train_sizes, train_scores_mean, label = "Training accuracy")
        plt.plot(train_sizes, valid_scores_mean, label = "Validation accuracy")
        plt.legend()
        plt.xlabel("Train Size")
        plt.ylabel("Score")
        #plt.savefig('validation.pdf')
        plt.show()
        
        
if __name__ == "__main__":
    
  
    #for decison tree uncomment following and run
    #model = Model()
    #model.curve_plots()
    
    
    #for logistic regression uncomment following and run
    model = Model(model = "Logistic Regression", scale = True)
    model.curve_plots()

                
        
        
        
        
        
    
        
        
        
        
        
        