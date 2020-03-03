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
import plotly.express as px


@st.cache
def loadData():
	df = pd.read_csv("train-bank-campaign-data.csv")
	return df
	
# Basic and common preprocessing required for all the models.  
def preprocessing(df):
    # Drop columns that don't contain useful information for our predictions
    df.drop(columns = ['id_var', 'duration'], inplace = True)

	# Assign X (independent features) and y (dependent feature i.e. df['y'] column in dataset)
    X = df.drop(columns = 'y')
    y = df['y']

    # Drop highly correlated features
    corr_matrix = X.corr().abs()
    tri = np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool)
    upper = corr_matrix.where(tri)
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)] # Find index of feature columns with correlation greater than 90%
    X.drop(X[to_drop], axis = 1, inplace = True)

    # X and y has Categorical data hence needs Encoding
    X = hotEncoding(X)
    le = LabelEncoder()
    y = le.fit_transform(y)

	# 1. Splitting X,y into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1, inplace=True)
    return X_train, X_test, y_train, y_test, le

def hotEncoding(df):
    cat_data = df.select_dtypes(include=['object']).copy()
    cat_data_dummies = pd.get_dummies(df[cat_data.columns])
    df.drop(columns = cat_data.columns, inplace = True)
    return pd.concat([cat_data_dummies, df], axis=1)

@st.cache(suppress_st_warning=True)
def logisticRegression(X_train, X_test, y_train, y_test):
	log_regr = LogisticRegression(solver = 'lbfgs')
	log_regr.fit(X_train, y_train.ravel())
	y_pred = log_regr.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, log_regr

@st.cache(suppress_st_warning=True)
def randomForestClassifier(X_train, X_test, y_train, y_test):
	rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, rf

# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
	# Scalling the data before feeding it to the Neural Network.
	scaler = StandardScaler()  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)
	# Instantiate the Classifier and fit the model.
	nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	nn.fit(X_train, y_train)
	y_pred = nn.predict(X_test)
	score1 = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)
	
	return score1, report, nn

# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, knn

def main():
	st.title("Prediction of Marketing Campaign result using various Machine Learning Classification Algorithms")
	data = loadData()
	X_train, X_test, y_train, y_test, le = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")	
		st.write(data.tail(20))


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Logistic Regression", "Neural Network", "Random Forest", "K-Nearest Neighbours"])

	if(choose_model == "Logistic Regression"):
		score, report, lr = logisticRegression(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Logistic Regression model is: ")
		st.write(score,"%")
		st.text("Report of Logistic Regression model is: ")
		st.write(report)

	elif(choose_model == "Random Forest"):
		score, report, rf = randomForestClassifier(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Random Forest model is: ")
		st.write(score,"%")
		st.text("Report of Random Forest model is: ")
		st.write(report)

	elif(choose_model == "Neural Network"):
		score, report, nn = neuralNet(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Neural Network model is: ")
		st.write(score,"%")
		st.text("Report of Neural Network model is: ")
		st.write(report)

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, knn = Knn_Classifier(X_train, X_test, y_train, y_test)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")
		st.text("Report of K-Nearest Neighbour model is: ")
		st.write(report)

	if st.sidebar.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values"):
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
		pdays = st.sidebar.text_input("Enter the pdays: ")
		previous = st.sidebar.text_input("Enter the previous: ")
		poutcome = st.sidebar.selectbox("Select the poutcome: ", data['poutcome'].unique())
		emp_var_rate = st.sidebar.text_input("Enter the emp var rate: ")
		cons_price_idx = st.sidebar.text_input("Enter the cons price idx: ")
		cons_conf_idx = st.sidebar.text_input("Enter the cons conf idx: ")
		submit = st.sidebar.button('Predict')
		if submit:
			df = pd.DataFrame([[int(age),job,marital,education,default,housing,loan,contact,month,day_of_week,int(campaign),int(pdays),int(previous),poutcome,float(emp_var_rate),float(cons_price_idx),float(cons_conf_idx)]],
			columns = ['age',f'job_{job}',f'marital_{marital}',f'education_{education}',f'default_{default}',f'housing_{housing}',f'loan_{loan}',f'contact_{contact}',f'month_{month}',f'day_of_week_{day_of_week}','campaign','pdays','previous',f'poutcome_{poutcome}','emp.var.rate','cons.price.idx','cons.conf.idx'])
			df1 = X_train.copy()
			df = pd.concat([df1, df], axis=0)
			df = df.fillna(0)
			cat_data = df.select_dtypes(include=['object']).copy()
			for col in cat_data.columns:
				df[col] = np.where(df[col] == 0, 0, 1)
			user_prediction_data = np.array(df.iloc[-1].values).reshape(1,-1)
			st.write(user_prediction_data)

			if(choose_model == "Logistic Regression"):
				pred = lr.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value.
			
			elif(choose_model == "Neural Network"):
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				user_prediction_data = scaler.transform(user_prediction_data)	
				pred = nn.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 

			elif(choose_model == "K-Nearest Neighbours"):
				pred = knn.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 

			elif(choose_model == "Random Forest"):
				pred = rf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 

if __name__ == "__main__":
	main()

# Visualization Section

data = loadData()

campaign_yes = data.query("y == 'yes'")

campaign_no = data.query("y == 'no'")

choose_viz = st.sidebar.selectbox("Choose the Visualization", 
["NONE","Observations by Campaign", "Observations by Target", "Campaigns when Target is Yes", "Campaigns when Target is No"])

if(choose_viz == "Observations by Campaign"):
	fig = px.histogram(data['campaign'], x ='campaign')
	st.plotly_chart(fig)

elif(choose_viz == "Observations by Target"):
	fig = px.histogram(data['y'], x ='y')
	st.plotly_chart(fig)

elif(choose_viz == "Campaigns when Target is Yes"):
	fig = px.histogram(campaign_yes['campaign'], x ='campaign')
	st.plotly_chart(fig)

elif(choose_viz == "Campaigns when Target is No"):
	fig = px.histogram(campaign_no['campaign'], x ='campaign')
	st.plotly_chart(fig)