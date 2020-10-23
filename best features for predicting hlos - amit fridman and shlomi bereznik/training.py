from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import mean_squared_error,make_scorer
import torch
import utils
from torch import nn
from math import sqrt
from sklearn.svm import SVC
import pandas as pd
import numpy as np

"""
inputs: y_true- the true labels
        y_pred- the labels predicted by the model
outputs: (-1)* RMSE value for the model
"""
def my_custom_loss_func(y_true, y_pred):
    diff = np.mean((y_true - y_pred)**2)
    return (-1)*sqrt(diff)
mse=make_scorer(my_custom_loss_func,greater_is_better=True) #creating a scorer from the rmse function

"""
inputs: test_dataloader- the dataloader for the test data
        model- the neural network model for evaluation
        test- the test data itself
outputs: the model rmse on the test data
"""
def evaluate(test_dataloader, model, test):
    floss = nn.MSELoss()
    acc = 0
    loss_avg = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            words_idx_tensor, labels = input_data
            labels=labels.type(torch.FloatTensor)
            outputs = model(words_idx_tensor.type(torch.FloatTensor))
            predicted=outputs
            #print(predicted,labels)
            loss_avg += floss(outputs,labels)
            acc+=(outputs-labels)**2
        acc = torch.sqrt(acc / len(test))
    return acc, loss_avg / len(test)


"""
inputs: features_list- matrix where each row is the list of best features according to each of the models
        k- the amount of features to take from each list
outputs: return a list of the features that are in at least one of the row of features_list matrix in the first k columns
"""
def best_features(features_list,k):
    features_scores={}
    for j in range (len(features_list)):
        print('j',[X.columns[features_list[j][i]] for i in range(k)])
        for feature in features_list[j][:k]:
            if feature in ['religion','ethnicity']:
                continue
            if feature not in features_scores:
                features_scores[feature]=1
                for i in range(j+1,len(features_list)):
                    features_scores[feature]+=1
    return sorted(features_scores.items(),key=lambda item: item[1],reverse=True)


"""
inputs: coef- list of importance scores for the features
        feature_list- list of all the features
outputs: feature list in decreasing order of importance.
"""
def feature_importance(coef,feature_list):
    features_dict={}
    #print(feature_list)
    #print(len(coef),len(feature_list))
    for i,feature in enumerate(feature_list):
        features_dict[i]=coef[i]
    ordered_features=sorted(features_dict.items(),key=lambda item: item[1],reverse=True)
    return  [a[0] for a in ordered_features]


"""
inputs: X- the test data as a tensor
        layers- list of the neural network model's layers
        _L- amount of layers in the model
outputs: the lrp scores for each of the features
"""
def Lrp(X,_layers,_L):
    A = [X] + [None] * _L  # initiliza the values of the activation layers
    for l in range(_L):
        A[l + 1] = _layers[l].forward(A[l])  # we fill the activation values of matrix
    chosen_label = A[_L].argmax().item()
    T = 1 # \\this is the mask/indicator which shows who is the true label.
    R = [None] * _L + [(A[-1] * T).data]  # we initialize the relavance by taking the last layer [(A[-1] * T).data]

    for l in range(0, _L)[::-1]:
        #the conditions holds for all layers except for the relu layer
        A[l] = (A[l].data).requires_grad_(True)
        #below incr and rho are help functions
        incr = lambda z: z + 1e-9
        rho = lambda p: p
        #Below we perform four steps for calculate the relevance scores for each layer.
        #The alorithm is iterative, we use the current relance scores layer to calculate the relevance scores
        #of the preavious layer
        z = incr(utils.newlayer(_layers[l], rho).forward(A[l]))  # step 1
        s = (R[l + 1] / z).data
       # print((z*s).sum())# )step 2
        (z * s).sum().backward()
        c = A[l].grad  # step 3
        R[l] = (A[l] * c).data  # step 4
    #print(R)
    return R[0]


#defintiion of the neural network model
class losnet(nn.Module):
    def __init__(self,features_num):
        super(losnet,self).__init__()
        self.Linear1=nn.Linear(features_num,20)
        self.relu=nn.ReLU()
        #self.Linear2=nn.Linear(40,20)
        self.Linear3=nn.Linear(20,1)

    def forward(self, x):
        x=self.Linear1(x)
        x=self.relu(x)
        x=self.Linear3(x)
        return x


if __name__ == '__main__':
    #finishing preprocessing of the data
    features_list=[]
    pd.set_option('display.max_columns', None)
    data=pd.read_csv('data.csv')

    data=data.drop(['map','MapApacheIIValue','HCO3Score','MapApacheIIScore','FiO2','FiO2_ApacheIV','Bilirubin_ApacheIV','los','last_wardid','last_careunit','subject_id','hadm_id'],axis=1)
    print(data['hlos'].describe())
    for col in data.columns:
        if data[col].dtype ==np.dtype('object'):
            data[col]=data[col].str.extract('(\d+)', expand=False)

    print(data.isnull().sum())
    data = data.fillna(-1)
    data = data[data['hlos']>=2]
    y = data['hlos']
    X=data.drop(['hlos','admittime'],axis=1)

    x=MinMaxScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    #traning and evaluating the neural network model
    train_data=[(torch.tensor(X_train[i].astype('float64'),requires_grad=True),y_train.values[i].astype('float64')) for i in range(len(X_train))]
    test_data = [(torch.tensor(X_test[i].astype('float64'),requires_grad=True), y_test.values[i].astype('float64')) for i in range(len(X_test))]

    train_dataloader=DataLoader(train_data,batch_size=100,shuffle=True)
    test_dataloader=DataLoader(test_data)
    floss=nn.MSELoss()

    model=losnet(len(X.columns))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.05)
    epochs=20

    accuracy_list = []
    loss_list = []
    test_loss = []
    acc_test = []
    max_acc = 0

    for epoch in range(epochs):
        acc = 0  # to keep track of accuracy
        loss1 = 0  # To keep track of the loss value
        i = 0
        model.train()
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, labels = input_data
            outputs = model(words_idx_tensor.type(torch.FloatTensor))
            predicted=outputs.view(outputs.shape[0])
            labels=labels.type(torch.FloatTensor)
            loss = floss(predicted, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            acc += ((predicted-labels)**2).sum().item()
            loss1 += loss

        loss = loss1 / len(train_data)
        acc = np.sqrt(acc / len(train_data))
        loss_list.append(float(loss))
        accuracy_list.append(float(acc))
        test_acc, loss_test = evaluate(test_dataloader, model, test_data)
        test_loss.append(loss_test)
        acc_test.append(test_acc)
        e_interval = i
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
                                                                                      np.mean(loss_list[-e_interval:]),
                                                                                      np.mean(
                                                                                          accuracy_list[-e_interval:]),
                                                                                      test_acc))
    layers = [module for module in model.modules()][1:]#net of the layers list

    L = len(layers)#Number of the layers in the trained net

    lrp_dataloader=DataLoader(test_data,batch_size=len(test_data))
    for batch in lrp_dataloader:
        document,true_label=batch
        x=document.type(torch.FloatTensor)
        print(x.shape)
        #We calculate the relevance scorss for the feature in the dataset
        relevance_score_lrp=torch.sum(Lrp(x,layers,L),dim=1)
        print(len(relevance_score_lrp))
        features_list.append(feature_importance(relevance_score_lrp,X.columns))

    #training and evaluationg a Linear Regression model
    model=LinearRegression()
    scores=cross_val_score(model,X_train,y_train,cv=5,scoring=mse)
    print('cross_val_scores',scores)
    model=model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('test_scores',my_custom_loss_func(y_test,y_pred))
    features_list.append(feature_importance(model.coef_,X.columns))

    #training and evaluating a Random Forest model
    model=RandomForestRegressor()
    grid={
        'n_estimators':[50,100,200],
        'min_samples_split':[2,4,6,8,10]
    }
    s = GridSearchCV(estimator=model, param_grid=grid,scoring=mse)
    s.fit(X_train, y_train)
    best_rf = s.best_estimator_
    print('croos_val_score',s.best_score_)
    y_pred=best_rf.predict(X_test)
    print('test_scores',my_custom_loss_func(y_test,y_pred))
    print('best_params',s.best_params_)
    features_list.append(feature_importance(best_rf.feature_importances_,X.columns))

    #training and evaluating a GBT model
    model = GradientBoostingRegressor()
    grid = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [50, 100, 200],

    }
    s = GridSearchCV(estimator=model, param_grid=grid,scoring=mse)
    s.fit(X_train, y_train)
    best_rf = s.best_estimator_
    print('croos_val_score',s.best_score_)
    y_pred = best_rf.predict(X_test)
    print('test_scores', my_custom_loss_func(y_test, y_pred))
    print('best_params',s.best_params_)
    features_list.append(feature_importance(best_rf.feature_importances_, X.columns))


    #training a GBT model only with the 20 most importance features of each of the models
    best_features_10=[a[0] for a in best_features(features_list,10)]
    print([X.columns[best_features_10[i]] for i in range(len(best_features_10))])
    X_train_10=X_train[:,best_features_10]
    model = GradientBoostingRegressor()
    grid = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [50, 100, 200],

    }
    s = GridSearchCV(estimator=model, param_grid=grid,scoring=mse)
    s.fit(X_train_10, y_train)
    best_rf = s.best_estimator_
    print('croos_val_score',s.best_score_)
    y_pred = best_rf.predict(X_test[:,best_features_10])
    print('test_scores', my_custom_loss_func(y_test, y_pred))
    print('best_params',s.best_params_)