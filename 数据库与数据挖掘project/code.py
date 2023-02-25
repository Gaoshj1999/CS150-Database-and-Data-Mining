# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:16:42 2021

@author: duola
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 04:50:50 2021

@author: duola
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random
"""
/////////////////////////////////////////////
"""
#对train进行data cleaning和调整
train_filepath = 'train.csv'
traindata = pd.read_table(train_filepath)
dfc=traindata.copy()
values = {'KC(Default)':0,'Opportunity(Default)':0}
dfc.fillna(value=values,inplace=True)
Opportunity= dfc['Opportunity(Default)']
dfc.drop(labels=['Opportunity(Default)'], axis=1,inplace = True)
dfc.insert(0, 'Opportunity(Default)', Opportunity)
Kc= dfc['KC(Default)']
dfc.drop(labels=['KC(Default)'], axis=1,inplace = True)
dfc.insert(0, 'KC(Default)', Kc)
"""
/////////////////////////////////////////////
"""
#调整kc和op的值
def OP(x):
    matrix=str(x).split('~~')
    mini=int(matrix[0])
    if len(matrix)>1:
        for i in range(len(matrix)):      
            if mini>int(matrix[i]):
                mini=int(matrix[i])  
    return mini
def KC(x):
    if x==0:
        return 0
    else:
        return 1
dfc['Opportunity(Default)'] = dfc['Opportunity(Default)'].apply(lambda x: OP(x)) 
dfc['KC(Default)'] = dfc['KC(Default)'].apply(lambda x: KC(x)) 
"""
/////////////////////////////////////////////
"""
##对test进行data cleaning和调整
test_filepath = 'test.csv'
testdata = pd.read_table(test_filepath)
dfct=testdata.copy()
values = {'KC(Default)':0,'Opportunity(Default)':0,'Correct First Attempt':'None'}
dfct.fillna(value=values,inplace=True)
Opportunity= dfct['Opportunity(Default)']
dfct.drop(labels=['Opportunity(Default)'], axis=1,inplace = True)
dfct.insert(0, 'Opportunity(Default)', Opportunity)
Kc= dfct['KC(Default)']
dfct.drop(labels=['KC(Default)'], axis=1,inplace = True)
dfct.insert(0, 'KC(Default)', Kc)
dfct['Opportunity(Default)'] = dfct['Opportunity(Default)'].apply(lambda x: OP(x)) 
dfct['KC(Default)'] = dfct['KC(Default)'].apply(lambda x: KC(x)) 
"""
/////////////////////////////////////////////
"""
#对Anon_Student_Id进行编码和配对
#编码
Anon_Student_Id=pd.factorize(dfc['Anon Student Id'])
Anon_Student_Idt=pd.factorize(dfct['Anon Student Id'])
checkpoint_AI=[]
maxindex_AI=max(len(Anon_Student_Id[1])-1,len(Anon_Student_Idt[1])-1)
#对新增部分进行重新编码
for i in range(len(Anon_Student_Idt[0])-1):
    checkpoint_AI.append(False)
for cur_index in range(len(Anon_Student_Idt[1])-1):
    if Anon_Student_Idt[1][cur_index] not in Anon_Student_Id[1]:
        for i in range(len(Anon_Student_Idt[0])-1):
            if Anon_Student_Idt[0][i]==cur_index:
                if checkpoint_AI[i]==False:
                    Anon_Student_Idt[0][i]=maxindex_AI+cur_index+1
                    checkpoint_AI[i]=True  
#对重叠部分进行配对
for cur_index in range(len(Anon_Student_Idt[1])-1):
    if Anon_Student_Idt[1][cur_index] in Anon_Student_Id[1]:
        true_index=-1
        for i in range(len(Anon_Student_Id[1])-1):
            if Anon_Student_Id[1][i]==Anon_Student_Idt[1][cur_index]:
                true_index=i
        for i in range(len(Anon_Student_Idt[0])-1):
            if Anon_Student_Idt[0][i]==cur_index:
                if checkpoint_AI[i]==False:
                    Anon_Student_Idt[0][i]=true_index
                    checkpoint_AI[i]=True
dfc['Anon Student Id']=Anon_Student_Id[0]
dfct['Anon Student Id']=Anon_Student_Idt[0]
"""
/////////////////////////////////////////////
"""
#对Problem_Hierarchy进行编码和配对
#编码
Problem_Hierarchy=pd.factorize(dfc['Problem Hierarchy'])
Problem_Hierarchyt=pd.factorize(dfct['Problem Hierarchy'])
checkpoint_PH=[]
maxindex_PH=max(len(Problem_Hierarchy[1])-1,len(Problem_Hierarchyt[1])-1)
#对新增部分进行重新编码
for i in range(len(Problem_Hierarchyt[0])-1):
    checkpoint_PH.append(False)
for cur_index in range(len(Problem_Hierarchyt[1])-1):
    if Problem_Hierarchyt[1][cur_index] not in Problem_Hierarchy[1]:
        for i in range(len(Problem_Hierarchyt[0])-1):
            if Problem_Hierarchyt[0][i]==cur_index:
                if checkpoint_PH[i]==False:
                    Problem_Hierarchyt[0][i]=maxindex_PH+cur_index+1
                    checkpoint_PH[i]=True  
#对重叠部分进行配对
for cur_index in range(len(Problem_Hierarchyt[1])-1):
    if Problem_Hierarchyt[1][cur_index] in Problem_Hierarchy[1]:
        true_index=-1
        for i in range(len(Problem_Hierarchy[1])-1):
            if Problem_Hierarchy[1][i]==Problem_Hierarchyt[1][cur_index]:
                true_index=i
        for i in range(len(Problem_Hierarchyt[0])-1):
            if Problem_Hierarchyt[0][i]==cur_index:
                if checkpoint_PH[i]==False:
                    Problem_Hierarchyt[0][i]=true_index
                    checkpoint_PH[i]=True
dfc['Problem Hierarchy']=Problem_Hierarchy[0]
dfct['Problem Hierarchy']=Problem_Hierarchyt[0]
"""
/////////////////////////////////////////////
"""
#对Problem_Name进行编码和配对
#编码
Problem_Name=pd.factorize(dfc['Problem Name'])
Problem_Namet=pd.factorize(dfct['Problem Name'])
checkpoint_PN=[]
maxindex_PN=max(len(Problem_Name[1])-1,len(Problem_Namet[1])-1)
#对新增部分进行重新编码
for i in range(len(Problem_Namet[0])-1):
    checkpoint_PN.append(False)
for cur_index in range(len(Problem_Namet[1])-1):
    if Problem_Namet[1][cur_index] not in Problem_Name[1]:
        for i in range(len(Problem_Namet[0])-1):
            if Problem_Namet[0][i]==cur_index:
                if checkpoint_PN[i]==False:
                    Problem_Namet[0][i]=maxindex_PN+cur_index+1
                    checkpoint_PN[i]=True  
#对重叠部分进行配对
for cur_index in range(len(Problem_Namet[1])-1):
    if Problem_Namet[1][cur_index] in Problem_Name[1]:
        true_index=-1
        for i in range(len(Problem_Name[1])-1):
            if Problem_Name[1][i]==Problem_Namet[1][cur_index]:
                true_index=i
        for i in range(len(Problem_Namet[0])-1):
            if Problem_Namet[0][i]==cur_index:
                if checkpoint_PN[i]==False:
                    Problem_Namet[0][i]=true_index
                    checkpoint_PN[i]=True
dfc['Problem Name']=Problem_Name[0]
dfct['Problem Name']=Problem_Namet[0]
"""
/////////////////////////////////////////////
"""
#对Step_Name进行编码和配对
#编码
Step_Name=pd.factorize(dfc['Step Name'])
Step_Namet=pd.factorize(dfct['Step Name'])
checkpoint_SN=[]
#对新增部分进行重新编码
for i in range(len(Step_Namet[0])-1):
    checkpoint_SN.append(False)
#对重叠部分进行配对
for cur_index in range(len(Step_Namet[1])-1):
    if Step_Namet[1][cur_index] in Step_Name[1]:
        true_index=-1
        for i in range(len(Step_Name[1])-1):
            if Step_Name[1][i]==Step_Namet[1][cur_index]:
                true_index=i
        for i in range(len(Step_Namet[0])-1):
            if Step_Namet[0][i]==cur_index:
                if checkpoint_SN[i]==False:
                    Step_Namet[0][i]=true_index
                    checkpoint_SN[i]=True
dfc['Step Name']=Step_Name[0]
dfct['Step Name']=Step_Namet[0]
"""
/////////////////////////////////////////////
"""
#对train删去无用的列并转化为feature
del dfc['Row']
del dfc['Step Start Time']
del dfc['First Transaction Time']
del dfc['Correct Transaction Time']
del dfc['Step End Time']
del dfc['Step Duration (sec)']
del dfc['Correct Step Duration (sec)']
del dfc['Error Step Duration (sec)']
del dfc['Incorrects']
del dfc['Hints']
del dfc['Corrects']
dfc=dfc.values
"""
/////////////////////////////////////////////
"""
#对test删去无用的列并转化为feature,并将test和predict部分拆开
del dfct['Row']
del dfct['Step Start Time']
del dfct['First Transaction Time']
del dfct['Correct Transaction Time']
del dfct['Step End Time']
del dfct['Step Duration (sec)']
del dfct['Correct Step Duration (sec)']
del dfct['Error Step Duration (sec)']
del dfct['Incorrects']
del dfct['Hints']
del dfct['Corrects']
dfct=dfct.values
test=[]
predict=[]
for i in range(len(dfct)):
    if dfct[i][len(dfct[i])-1]=='None':
        predict.append(dfct[i])
    else:
        test.append(dfct[i])
"""
/////////////////////////////////////////////
"""
#引入logistic regression模型进行训练和预测
from sklearn.linear_model import LogisticRegression
def colicSklearn(filetrain,filetest,filepredict):
    dfc=filetrain.copy()
    dfct=filetest.copy()
    dfcp=filepredict.copy()
    trainingSet = []
    trainingLabels = []
    for i in range(len(dfc)):
        linearr=[]
        for j in range(len(dfc[i])-1):
            linearr.append(dfc[i][j])
        trainingSet.append(linearr)
        trainingLabels.append(dfc[i][len(dfc[i])-1])
    testSet =  []
    testLabels = []
    for i in range(len(dfct)):
        linearr=[]
        for j in range(len(dfct[i])-1):
            linearr.append(dfct[i][j])
        testSet.append(linearr)
        testLabels.append(dfct[i][len(dfct[i])-1])
    classifier = LogisticRegression(solver='liblinear',max_iter=1000).fit(trainingSet,trainingLabels)
    test_accurcy = classifier.score(testSet,testLabels)*100
    #test的正确率以及训练得出的参数
    print("正确率为%s%%"%test_accurcy)
    print("Coefficients:",classifier.coef_)
    predictSet=[]
    for i in range(len(dfcp)):
        linearr=[]
        for j in range(len(dfcp[i])-1):
            linearr.append(dfcp[i][j])
        predictSet.append(linearr)
    #预测的标签
    print(classifier.predict(predictSet))
   
if __name__ == '__main__':
    colicSklearn(dfc,test,predict)