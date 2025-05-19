import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit_stroke import clf,X_split,Y_split,X_df
np.random.seed(69420)

median_age = X_df['Age'].median()
median_bmi=X_df['BMI'].median()
median_glucose=X_df['Avg_Glucose'].median()

def bins_2(X,Y,feature):   
    col = X_df.columns.get_loc(feature)
    X_0 = X[X[:, col] == 0]   # Select 0
    Y_0 = Y[X[:, col] == 0]   

    X_1 = X[X[:, col] == 1]  # Select 1
    Y_1 = Y[X[:, col] == 1]  
    return X_1, Y_1, X_0, Y_0, col

def bins_3(X,Y,feature1,feature2,feature3):
    col1 = X_df.columns.get_loc(feature1)
    col2 = X_df.columns.get_loc(feature2)
    col3 = X_df.columns.get_loc(feature3)
    X_0 = X[X[:, col1] == 1]   # Select 0
    Y_0 = Y[X[:, col1] == 1]   

    X_1 = X[X[:, col2] == 1]  # Select 1
    Y_1 = Y[X[:, col2] == 1]
    
    X_2 = X[X[:, col3] == 1]  # Select 2
    Y_2 = Y[X[:, col3] == 1] 
     
    return X_2,Y_2,X_1, Y_1, X_0, Y_0, col1,col2,col3

def bins_con(X,Y,feature,threshold):
    col = X_df.columns.get_loc(feature)
    X_0 = X[X[:, col] <threshold]   # Select under threshold
    Y_0 = Y[X[:, col] <threshold]   

    X_1 = X[X[:, col] >=threshold]  # Select above threshold
    Y_1 = Y[X[:, col] >=threshold]  
    return X_1, Y_1, X_0, Y_0, col

def scp_conformal_quantile(X_cali, y_cali, alpha):
    phats = clf.predict_proba(X_cali)
    scores = 1 - phats[np.arange(len(y_cali)), y_cali]
    qhat = np.quantile(scores, (1-alpha)*(1+1/(len(y_cali))))
    return qhat

def scp_compute_C_feature_2_bins(X_test, y_test, X_cali, y_cali, alpha, feature, categories=np.array([0, 1])):
    qhat = scp_conformal_quantile(X_cali, y_cali, alpha)
    n = len(X_test)
    X_1_t, Y_1_t, X_0_t, Y_1_t,col=bins_2(X_test,y_test,feature)
    n_0=len(X_0_t)
    n_1=len(X_1_t)
    prediction_sets = np.zeros(n, dtype = object) 
    phats = clf.predict_proba(X_test)
    hits = 0
    hits_0 = 0
    hits_1 = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
            if X_test[i][col]==0:
                    hits_0+=1
            elif X_test[i][col]==1:
                    hits_1+=1
    emp_coverage_all = 1/n * hits
    emp_coverage_0=1/n_0*hits_0
    emp_coverage_1=1/n_1*hits_1
    emp_coverage=np.array([emp_coverage_0,emp_coverage_1,emp_coverage_all])
    return emp_coverage

def scp_compute_C_feature_3_bins(X_test, y_test, X_cali, y_cali, alpha, feature1,feature2,feature3, categories=np.array([0, 1])):
    qhat = scp_conformal_quantile(X_cali, y_cali, alpha)
    n = len(X_test)
    X_2_t, Y_2_t, X_1_t, Y_1_t, X_0_t, Y_0_t, col1, col2, col3 = bins_3(X_test, y_test, feature1, feature2, feature3)
    n_0=len(X_0_t)
    n_1=len(X_1_t)
    n_2=len(X_2_t)
    prediction_sets = np.zeros(n, dtype = object) 
    phats = clf.predict_proba(X_test)
    hits = 0
    hits_0 = 0
    hits_1 = 0
    hits_2 = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
            if X_test[i][col1]==1:
                    hits_0+=1
            elif X_test[i][col2]==1:
                    hits_1+=1        
            elif X_test[i][col3]==1:
                    hits_2+=1
    emp_coverage_all = hits *1/ n
    emp_coverage_0 = hits_0 *1/ n_0
    emp_coverage_1 = hits_1 *1/ n_1
    emp_coverage_2 = hits_2 *1/ n_2
    emp_coverage=np.array([emp_coverage_0,emp_coverage_1,emp_coverage_2,emp_coverage_all])
    return emp_coverage

def scp_compute_C_feature_con(X_test, y_test, X_cali, y_cali, alpha, feature,threshold, categories=np.array([0, 1])):
    qhat = scp_conformal_quantile(X_cali, y_cali, alpha)
    n = len(X_test)
    X_O_t, Y_O_t, X_U_t, Y_U_t,col=bins_con(X_test,y_test,feature,threshold)
    n_U=len(X_U_t)
    n_O=len(X_O_t)
    prediction_sets = np.zeros(n, dtype = object) 
    phats = clf.predict_proba(X_test)
    hits = 0
    hits_U = 0
    hits_O = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
            if X_test[i][col]<threshold:
                    hits_U+=1
            elif X_test[i][col]>=threshold:
                    hits_O+=1
    emp_coverage_all = 1/n * hits
    emp_coverage_U=1/n_U*hits_U
    emp_coverage_O=1/n_O*hits_O
    emp_coverage=np.array([emp_coverage_U,emp_coverage_O,emp_coverage_all])
    return emp_coverage


def plot_alpha_motivation_feature(alphas,n_runs):
        emp_cov_age_less_69=np.zeros_like(alphas)
        emp_cov_age_69=np.zeros_like(alphas)
        emp_cov_Hypertension_0=np.zeros_like(alphas)
        emp_cov_Hypertension_1=np.zeros_like(alphas)
        emp_cov_HD_0=np.zeros_like(alphas)
        emp_cov_HD_1=np.zeros_like(alphas)
        emp_cov_BMI_U=np.zeros_like(alphas)
        emp_cov_BMI_O=np.zeros_like(alphas)
        emp_cov_Glucose_less_100=np.zeros_like(alphas)
        emp_cov_Glucose_100=np.zeros_like(alphas)
        emp_cov_Diabetes_0=np.zeros_like(alphas)
        emp_cov_Diabetes_1=np.zeros_like(alphas)
        emp_cov_Gender_0=np.zeros_like(alphas)
        emp_cov_Gender_1=np.zeros_like(alphas)
        emp_cov_SES_0=np.zeros_like(alphas)
        emp_cov_SES_1=np.zeros_like(alphas)
        emp_cov_SES_2=np.zeros_like(alphas)
        emp_cov_smoking_0=np.zeros_like(alphas)
        emp_cov_smoking_1=np.zeros_like(alphas)
        emp_cov_smoking_2=np.zeros_like(alphas)
        for k in range(n_runs):
                print(f'round {k+1} out of {n_runs}')
                np.random.seed(k + 69)
                X_cali, X_test, Y_cali, Y_test= train_test_split(X_split,Y_split,stratify=Y_split,test_size=0.5,shuffle=True)
                for i, alpha in enumerate(alphas):
                         emp_coverage_Hypertension = scp_compute_C_feature_2_bins(X_test, Y_test, X_cali, Y_cali, alpha, "Hypertension")
                         emp_cov_Hypertension_0[i]+=emp_coverage_Hypertension[0]
                         emp_cov_Hypertension_1[i]+=emp_coverage_Hypertension[1]
                         
                         emp_coverage_HD = scp_compute_C_feature_2_bins(X_test, Y_test, X_cali, Y_cali, alpha, "Heart_Disease")
                         emp_cov_HD_0[i]+=emp_coverage_HD[0]
                         emp_cov_HD_1[i]+=emp_coverage_HD[1]
                         
                         emp_coverage_gender = scp_compute_C_feature_2_bins(X_test, Y_test, X_cali, Y_cali, alpha, "Gender_Female")
                         emp_cov_Gender_0[i]+=emp_coverage_gender[0]
                         emp_cov_Gender_1[i]+=emp_coverage_gender[1]
                         
                         emp_coverage_diabetes=scp_compute_C_feature_2_bins(X_test, Y_test, X_cali, Y_cali, alpha, "Diabetes")
                         emp_cov_Diabetes_0[i]+=emp_coverage_diabetes[0]
                         emp_cov_Diabetes_1[i]+=emp_coverage_diabetes[1]
                         
                         emp_coverage_age=scp_compute_C_feature_con(X_test, Y_test, X_cali, Y_cali, alpha, "Age",median_age)
                         emp_cov_age_less_69[i]+=emp_coverage_age[0]
                         emp_cov_age_69[i]+=emp_coverage_age[1]
                         
                         
                         emp_coverage_bmi=scp_compute_C_feature_con(X_test, Y_test, X_cali, Y_cali, alpha, "BMI",median_bmi)
                         emp_cov_BMI_U[i]+=emp_coverage_bmi[0] 
                         emp_cov_BMI_O[i]+=emp_coverage_bmi[1]
                         
                         emp_coverage_glucose=scp_compute_C_feature_con(X_test, Y_test, X_cali, Y_cali, alpha, "Avg_Glucose",median_glucose)
                         emp_cov_Glucose_less_100[i]+=emp_coverage_glucose[0] 
                         emp_cov_Glucose_100[i]+=emp_coverage_glucose[1]
                         
                         emp_coverage_SES=scp_compute_C_feature_3_bins(X_test, Y_test, X_cali, Y_cali, alpha, "SES_High","SES_Medium","SES_Low")
                         emp_cov_SES_0[i]+=emp_coverage_SES[0]
                         emp_cov_SES_1[i]+=emp_coverage_SES[1]
                         emp_cov_SES_2[i]+=emp_coverage_SES[2]
                         
                         emp_coverage_smoke=scp_compute_C_feature_3_bins(X_test, Y_test, X_cali, Y_cali, alpha, "Smoking_Status_Current","Smoking_Status_Former","Smoking_Status_Never")
                         emp_cov_smoking_0[i]+=emp_coverage_smoke[0]
                         emp_cov_smoking_1[i]+=emp_coverage_smoke[1]
                         emp_cov_smoking_2[i]+=emp_coverage_smoke[2] 
                           
        emp_cov_age_less_69_r = emp_cov_age_less_69/n_runs
        emp_cov_age_69_r = emp_cov_age_69/n_runs
        emp_cov_Hypertension_0_r = emp_cov_Hypertension_0/n_runs
        emp_cov_Hypertension_1_r = emp_cov_Hypertension_1/n_runs
        emp_cov_HD_0_r = emp_cov_HD_0/n_runs
        emp_cov_HD_1_r = emp_cov_HD_1/n_runs
        emp_cov_BMI_U_r = emp_cov_BMI_U/n_runs
        emp_cov_BMI_O_r = emp_cov_BMI_O/n_runs
        emp_cov_Glucose_less_100_r = emp_cov_Glucose_less_100/n_runs
        emp_cov_Glucose_100_r = emp_cov_Glucose_100/n_runs
        emp_cov_Diabetes_0_r = emp_cov_Diabetes_0/n_runs
        emp_cov_Diabetes_1_r = emp_cov_Diabetes_1/n_runs
        emp_cov_Gender_0_r = emp_cov_Gender_0/n_runs
        emp_cov_Gender_1_r = emp_cov_Gender_1/n_runs
        emp_cov_SES_0_r = emp_cov_SES_0/n_runs
        emp_cov_SES_1_r = emp_cov_SES_1/n_runs
        emp_cov_SES_2_r = emp_cov_SES_2/n_runs
        emp_cov_smoking_0_r = emp_cov_smoking_0/n_runs
        emp_cov_smoking_1_r = emp_cov_smoking_1/n_runs
        emp_cov_smoking_2_r = emp_cov_smoking_2/n_runs
        
        fig = plt.figure(figsize=(15, 12))

        fig.suptitle("Empirical Coverage", fontsize=20, y=0.98)
        
        plt.subplot(3, 3, 1)
        plt.plot(alphas,emp_cov_Hypertension_0_r,label='No Hypertension')
        plt.plot(alphas,emp_cov_Hypertension_1_r,label='Hypertension')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Hypertension')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 2)
        plt.plot(alphas,emp_cov_HD_0_r,label='No Heart Disease')
        plt.plot(alphas,emp_cov_HD_1_r,label='Heart Disease')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Heart Disease')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 3)
        plt.plot(alphas,emp_cov_Diabetes_0_r,label='No Diabetes')
        plt.plot(alphas,emp_cov_Diabetes_1_r,label='Diabetes')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Diabetes')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        # Second row
        plt.subplot(3, 3, 4)
        plt.plot(alphas,emp_cov_Gender_0_r,label='Male')
        plt.plot(alphas,emp_cov_Gender_1_r,label='Female')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Gender')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 5)
        plt.plot(alphas,emp_cov_smoking_0_r,label='Current Smoker')
        plt.plot(alphas,emp_cov_smoking_1_r,label='Former Smoker')
        plt.plot(alphas,emp_cov_smoking_2_r,label='Never Smoked')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Smoking Status')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 6)
        plt.plot(alphas,emp_cov_SES_0_r,label='Low')
        plt.plot(alphas,emp_cov_SES_1_r,label='Medium')
        plt.plot(alphas,emp_cov_SES_2_r,label='High')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Socioeconomic Status')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        # Third row
        plt.subplot(3, 3, 7)
        plt.plot(alphas,emp_cov_age_less_69_r,label=f'Under {int(np.floor(median_age))}')
        plt.plot(alphas,emp_cov_age_69_r,label=f'Over {int(np.floor(median_age))}')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Age')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 8)
        plt.plot(alphas,emp_cov_BMI_U_r,label=f'Under {int(np.floor(median_bmi))}')
        plt.plot(alphas,emp_cov_BMI_O_r,label=f'Over {int(np.floor(median_bmi))}')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('BMI')
        plt.xlabel(r'$\alpha$')

        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.subplot(3, 3, 9)
        plt.plot(alphas,emp_cov_Glucose_less_100_r,label=f'Under {int(np.floor(median_glucose))}')
        plt.plot(alphas,emp_cov_Glucose_100_r,label=f'Over {int(np.floor(median_glucose))}')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.title('Average Glucose Level')
        plt.xlabel(r'$\alpha$')
        plt.legend(loc = "upper right")
        plt.grid(True)

        plt.tight_layout(pad=3)
        plt.show()
        
alphas = np.linspace(0.01, 0.5, 150)
plot_alpha_motivation_feature(alphas,10)
