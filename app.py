import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import plotly.express as px


def loadData():
	X = pd.read_csv("features.csv")
	y = pd.read_csv("target.csv")['target']
	return X, y

def loadRawData():
	df = pd.read_csv("train-bank-campaign-data.csv")
	return df

def logisticRegression(X_train, y_train):
	log_regr = LogisticRegression(solver = 'lbfgs')
	scores = cross_val_score(log_regr, X_train, y_train, cv=5)
	log_regr.fit(X_train, y_train)
	return scores.mean(), log_regr

#@st.cache(suppress_st_warning=True)
def randomForestClassifier(X_train, y_train):
	# Instantiate the Classifier and fit the model.
	rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
	scores = cross_val_score(rf, X_train, y_train, cv=5)
	rf.fit(X_train, y_train)
	return scores.mean(), rf

# Training Neural Network for Classification.
#@st.cache(suppress_st_warning=True)
def neuralNet(X_train, y_train):
	# Scalling the data before feeding it to the Neural Network.
	scaler = StandardScaler()  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	# Instantiate the Classifier and fit the model.
	nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	scores = cross_val_score(nn, X_train, y_train, cv=5)
	nn.fit(X_train, y_train)
	return scores.mean(), nn

# Training KNN Classifier
#@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, y_train):
	# Instantiate the Classifier and fit the model.
	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(X_train, y_train)
	scores = cross_val_score(knn, X_train, y_train, cv=5)
	return scores.mean(), knn

def Print_Selected_Values(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx):
	st.write(f'Age: {age}')
	st.write(f'Job: {job}')
	st.write(f'Marital: {marital}')
	st.write(f'Education: {education}')
	st.write(f'Default: {default}')
	st.write(f'Housing: {housing}')
	st.write(f'Loan: {loan}')
	st.write(f'Contact: {contact}')
	st.write(f'Month: {month}')
	st.write(f'Day of week: {day_of_week}')
	st.write(f'Campaign: {campaign}')
	st.write(f'Previous: {previous}')
	st.write(f'Poutcome: {poutcome}')
	st.write(f'Emp var rate: {emp_var_rate}')
	st.write(f'Cons price idx: {cons_price_idx}')
	st.write(f'Cons conf idx: {cons_conf_idx}')

def main():
	st.title("Prediction of Marketing Campaign result using various Machine Learning Classification Algorithms")
	data = loadRawData()
	X_train, y_train = loadData()


	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")	
		st.write(data.tail(10))


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Logistic Regression", "Neural Network", "Random Forest", "K-Nearest Neighbours"])

	if(choose_model == "Logistic Regression"):
		score, lr = logisticRegression(X_train, y_train)
		st.text("Accuracy of Logistic Regression model is: ")
		st.write(score,"%")

	elif(choose_model == "Random Forest"):
		score, rf = randomForestClassifier(X_train, y_train)
		st.text("Accuracy of Random Forest model is: ")
		st.write(score,"%")

	elif(choose_model == "Neural Network"):
		score, nn = neuralNet(X_train, y_train)
		st.text("Accuracy of Neural Network model is: ")
		st.write(score,"%")

	elif(choose_model == "K-Nearest Neighbours"):
		score, knn = Knn_Classifier(X_train, y_train)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")

	if st.sidebar.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values"):
		choose_default_var = st.sidebar.selectbox("Fill variables", ["NONE","Positive sample", "Negative sample"])
		age = st.sidebar.text_input("Enter the age: ")
		job = st.sidebar.selectbox("Select the job: ", data['job'].unique())
		marital = st.sidebar.selectbox("Select the end station number: ", data['marital'].unique())
		education = st.sidebar.selectbox("Select the education: ", data['education'].unique())
		default = st.sidebar.selectbox("Select the default: ", data['default'].unique())
		housing = st.sidebar.selectbox("Select the housing: ", data['housing'].unique())
		loan = st.sidebar.selectbox("Select the loan: ", data['loan'].unique())
		contact = st.sidebar.selectbox("Select the contact: ", data['contact'].unique())
		month = st.sidebar.selectbox("Select the month: ", data['month'].unique())
		day_of_week = st.sidebar.selectbox("Select the day of week: ", data['day_of_week'].unique())
		campaign = st.sidebar.text_input("Enter the campaign: ")
		previous = st.sidebar.text_input("Enter the previous: ")
		poutcome = st.sidebar.selectbox("Select the poutcome: ", data['poutcome'].unique())
		emp_var_rate = st.sidebar.text_input("Enter the emp var rate: ")
		cons_price_idx = st.sidebar.text_input("Enter the cons price idx: ")
		cons_conf_idx = st.sidebar.text_input("Enter the cons conf idx: ")
		values_by_default = False
		
		if choose_default_var == "Positive sample":
			age = 38
			job = 'housemaid'	
			marital = 'divorced'
			education =	'high.school'
			default = 'no'
			housing = 'yes'
			loan = 'yes'
			contact = 'cellular'
			month =	'nov'
			day_of_week = 'thu'
			campaign = 1
			previous = 0
			poutcome = 'nonexistent'
			emp_var_rate = -1.1000			
			cons_price_idx = 94.7670
			cons_conf_idx = -50.8000
			values_by_default = True
		elif(choose_default_var == "Negative sample"):
			age = 57
			job = 'retired'	
			marital = 'married'
			education =	'professional.course'
			default = 'no'
			housing = 'yes'
			loan = 'no'
			contact = 'cellular'
			month =	'nov'
			day_of_week = 'thu'
			campaign = 6
			previous = 0
			poutcome = 'nonexistent'
			emp_var_rate = -1.1000			
			cons_price_idx = 94.7670
			cons_conf_idx = -50.8000
			values_by_default = True

		submit = st.sidebar.button('Predict')
		if submit:
			df = pd.DataFrame([[int(age),job,marital,education,default,housing,loan,contact,month,day_of_week,int(campaign),int(previous),poutcome,float(emp_var_rate),float(cons_price_idx),float(cons_conf_idx)]],
			columns = ['age',f'job_{job}',f'marital_{marital}',f'education_{education}',f'default_{default}',f'housing_{housing}',f'loan_{loan}',f'contact_{contact}',f'month_{month}',f'day_of_week_{day_of_week}','campaign','previous',f'poutcome_{poutcome}','emp.var.rate','cons.price.idx','cons.conf.idx'])
			df1 = X_train.copy()
			df = pd.concat([df1, df], axis=0)
			df = df.fillna(0)
			cat_data = df.select_dtypes(include=['object']).copy()
			for col in cat_data.columns:
				df[col] = np.where(df[col] == 0, 0, 1)
			user_prediction_data = np.array(df.iloc[-1].values).reshape(1,-1)

			if(choose_model == "Logistic Regression"):
				pred = lr.predict(user_prediction_data)
				st.write("The Predicted Class is: ", "NO" if pred[0] == 0  else "YES") # Inverse transform to get the original dependent value.
				if values_by_default:
					Print_Selected_Values(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx)
			
			elif(choose_model == "Neural Network"):
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				user_prediction_data = scaler.transform(user_prediction_data)	
				pred = nn.predict(user_prediction_data)
				st.write("The Predicted Class is: ", "NO" if pred[0] == 0  else "YES") # Inverse transform to get the original dependent value. 
				if values_by_default:
					Print_Selected_Values(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx)

			elif(choose_model == "K-Nearest Neighbours"):
				pred = knn.predict(user_prediction_data)
				st.write("The Predicted Class is: ", "NO" if pred[0] == 0  else "YES") # Inverse transform to get the original dependent value. 
				if values_by_default:
					Print_Selected_Values(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx)

			elif(choose_model == "Random Forest"):
				pred = rf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", "NO" if pred[0] == 0  else "YES") # Inverse transform to get the original dependent value. 
				if values_by_default:
					Print_Selected_Values(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx)

if __name__ == "__main__":
	main()

#Visualization Section

data = loadRawData()
cat_data = data.select_dtypes(include=['object']).copy()
numerical_data = data.select_dtypes(include=['int', 'float']).copy()
campaign_yes = cat_data.query("y == 'yes'")
campaign_no = cat_data.query("y == 'no'")

choose_viz = st.sidebar.selectbox("Choose the Visualization", 
["NONE", "Histograms - Categorical data", "Target Yes vs No"])

if(choose_viz == "Histograms - Categorical data"):
	for col in cat_data.columns:
		fig = px.histogram(data[col], x =col, title = f'{col}')
		st.plotly_chart(fig)

elif(choose_viz == "Target Yes vs No"):
	for col in cat_data.columns:
		fig = px.histogram(campaign_yes[col], x =col, title = f'{col}_yes')
		fig1 = px.histogram(campaign_no[col], x =col, title = f'{col}_no')
		st.plotly_chart(fig)
		st.plotly_chart(fig1)

elif(choose_viz == "Observations by Target"):
	fig = px.histogram(data['y'], x ='y')
	st.plotly_chart(fig)



