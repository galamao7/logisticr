import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np 
nhs_df = pd.io.parsers.read_csv("all.csv")


"""
This function is used to handle the missing values

Args:
    original: the original missing values
    new: all missing values are replaced with "NaN"
    
Returns:
    A new dataset with all missing values "NaN"
"""
def mv(data,original=[888888.0,88888.0,8888.0,888.0],new=np.NaN):
    newdata=data.replace(original, new)
    return newdata
nhs=mv(nhs_df)
#create an intercept for logistic regression
nhs['intercept']=1.0


# creat new variables
# aptp_ratio is calculated as the ratio of animal protein to total protein
# vptp_ratio is calculated as the ratio of vegetable protein to total protein
nhs["aptp_ratio"]=nhs["ani_prot"]/nhs["tot_prot_2"]
nhs["vptp_ratio"]=nhs["veg_prot"]/nhs["tot_prot_2"]


"""
This function is to create binary variables according to continuous values

Args:
    dataset:which dataset is being used 
    cate_var: the binary variable to be created
    var: the continuous variable we use to create binary variable
    threshold: specific value which is used to dichotomize the continuous variable
    
Returns:
    A new dataset which contains the binary variables we create
"""
    
def create_category_var(dataset,cate_var,var,threshold):
    dataset[cate_var] = (nhs[var] > threshold).astype(float).where(dataset[var].notnull())
    return dataset

# If overweight = 0, you will get results from subjects whose bmi<=25;
# If overweight = 1, you will get results from overweight subjects shose bmi > 25;
create_category_var(nhs,"overweight","bmi",25)

# If crp_cate = 0, you will get results from subjects whose crp<=3;
# If crp_cate = 1, you will get results from subjects whose crp>3;
create_category_var(nhs,'crp_cate','CRP',3)

#If hbp = 0, you will get results from subjects who are not high blood pressure patients.
#If hbp = 1, you will get results from subjects who with high blood pressure.
nhs['hbp'] = ((nhs['sbp'] > 140) |(nhs['dbp'] > 90)).astype(float).where(nhs['dbp'].notnull())


"""
This function is used to demonstrate descriptive statistics, run the logistic regression and plots of prediction value 

Args:
    dep: outcome variable. You can choose "overweight", "crp_cate", and "hbp"
    indep: independent variable
        'CRP' (inflammation marker)
        'bicarb'(serum bicarbonate)
        'phos'(serum phosphorus)
        'sbp'(systolic blood pressure)
        'dbp'(diastolic blood pressure)
        'bmi' (body mass index).
    var: the variable that you want to show in histogram
    pop_overweight: an argument that allows you to filter subjects by their bmi.
        If overweight = 0, you will get results from subjects whose bmi<=25;
        If overweight = 1, you will get results from overweight subjects shose bmi > 25;
        If overweight = 2, you will get results from all subjects.
    pop_hbp: "hbp" stands for high blood pressure. It helps you filter subjects by their blood pressure
        High blood pressure is definded as sbp>140 or dbp>90.
        If hbp = 0, you will get results from subjects who are not high blood pressure patients.
        If hbp = 1, you will get results from subjects who with high blood pressure.
        if hbp = 2, you will get results from all subjects.
    
Returns:
    1)An overview of the data
    2)Descriptive statistics (e.g., mean, std, mean, max)
    3)Histogram of different variables
    4)Output from logistic regression (e.g., parameter estimates, std, and p-value)
    5)Odds ratio of parameters and its 95% CI
    6)Plots of the relationship between prediction value and independent variable
"""
def logisticr(dep,indep,var="bmi",pop_overweight=2,pop_hbp=2):
    #filter the subjects by their bmi
    if pop_overweight == 0:
        nhs_ow = nhs[nhs["bmi"] <= 25]
    elif pop_overweight == 1:
        nhs_ow = nhs[nhs["bmi"] > 25]
    elif pop_overweight == 2:
        nhs_ow = nhs
    #filter the subjects by their blood pressure
    if pop_hbp == 0:
        nhs_hbp = nhs_ow[(nhs_ow['sbp'] < 140) & (nhs_ow['dbp'] < 90)]
    elif pop_hbp == 1:
        nhs_hbp = nhs_ow[(nhs_ow['sbp'] > 140) | (nhs_ow['dbp'] > 90)]
    elif pop_hbp == 2:
        nhs_hbp = nhs_ow
    # take a look at the dataset
    print nhs_hbp.head()
    # summarize the data    
    print nhs_hbp.describe()
    # plot all of the columns
    nhs_hbp[var].hist()
    pl.xlabel(var)         
    pl.title("Histogram of " + var)
    pl.show()
    # logistic regression
    logistic = sm.Logit(nhs_hbp[dep], nhs_hbp[['intercept',indep]],missing='drop') 
    # fit the model
    result = logistic.fit()
    print result.summary()
    #odds ratio and 95% CI
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    print np.exp(conf)
    # plot of prediction values
    nhs_hbp['predictions'] = result.predict(nhs_hbp[['intercept',indep]])
    pl.plot(nhs_hbp[indep], nhs_hbp['predictions'])
    pl.xlabel(indep)    
    pl.ylabel("P (" + dep + "=1)")        
    pl.title("Prob (" + dep + "=1) and " + indep)
    pl.show()

#test the function
logisticr("overweight","vptp_ratio",pop_hbp=1)
logisticr("CRP","vptp_ratio")
