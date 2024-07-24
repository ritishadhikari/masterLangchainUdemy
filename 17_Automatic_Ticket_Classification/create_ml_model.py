import streamlit as st
import admin_utils
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

if "cleanedData" not in st.session_state:
    st.session_state['cleanedData']=""
if "sentenceTrain" not in st.session_state:
    st.session_state['sentenceTrain']=""
if "sentenceTest" not in st.session_state:
    st.session_state['sentenceTest']=""
if "labelsTrain" not in st.session_state:
    st.session_state['labelsTrain']=""
if "labelsTest" not in st.session_state:
    st.session_state['labelsTest']=""
if "svmClassifier" not in st.session_state:
    st.session_state['svmClassifier']=""

st.title(body="Let's build our model...")

# Create tabs
tabTitles=['Data Preprocessing','Model Training','Model Evaluation','Save Model']
tabs=st.tabs(tabs=tabTitles)

# Adding Content to each tab

# Data Preprocessing Tab
with tabs[0]:
    st.header(body="Data Preprocessing")
    st.write("Here we preprocess the data...")

    data=st.file_uploader(label="Upload CSV FIle",type="csv")
    button=st.button(label="Load Data",key="data")

    if button:
        with st.spinner(text="Wait for it..."):
            ourData=admin_utils.readData(data=data)
            embeddings=admin_utils.createEmbeddingsLoadData()
            st.session_state['cleanedData']=admin_utils.createEmbeddings(
                df=ourData,
                embeddings=embeddings
                )
        st.success(body="Done")

# Model Training Tab
with tabs[1]:
    st.header(body="Model Training")
    st.write("Here we train the model...")
    button=st.button(label="Train the Model",key="model")

    if button:
        with st.spinner(text="Wait for it ..."):
            st.session_state["sentenceTrain"],st.session_state["sentenceTest"],\
            st.session_state["labelsTrain"],st.session_state["labelsTest"]=\
                sentTrain, sentTrain, labTrain, labTest=admin_utils.\
                splitTrainTestData(data=st.session_state["cleanedData"])
            
            st.session_state['svmClassifier']=make_pipeline(
                StandardScaler(),
                SVC(class_weight="balanced")
            )

            st.session_state['svmClassifier'].fit(
                st.session_state['sentenceTrain'],
                st.session_state['labelsTrain']
            )
        st.success(body="Model Training Done!")

# Model Evaluation Tab
with tabs[2]:
    st.header(body="Model Evaluation")
    st.write("Here we evaluate the model ...")
    button=st.button(label="Evaluate Model",key="Evaluation")

    if button:
        with st.spinner(text="Wait for it ..."):
            accuracyScore=admin_utils.getScore(
                svmClassifier= st.session_state['svmClassifier'],
                sentTest=st.session_state["sentenceTest"],
                labTest=st.session_state["labelsTest"]
                )
            st.success(body=f"Validation Accuracy is {100*accuracyScore}%")
            st.write("A sample run:")

            text="Rude Driver with scary driving"
            st.write(f"Our Issue: {text}")

            # Converting our TEXT to NUMBERICAL representation
            embeddings=admin_utils.createEmbeddingsLoadData()
            queryResult=embeddings.embed_query(text)

            result=st.session_state['svmClassifier'].predict([queryResult])[0]

            st.write(f"Department which it belongs to: {result}")
        st.success(body="Done!")

# Save Model Tab
with tabs[3]:
    st.header(body="Save Model")
    st.write("Here we save the model")

    button=st.button(label="Save Model",
                     key="Save")
    if button:
        with st.spinner(text="Wait for it ..."):
            joblib.dump(value=st.session_state['svmClassifier'],filename="modelSVM.pkl")
        st.success(body="Done !")