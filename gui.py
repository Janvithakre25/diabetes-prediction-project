import pickle
pickle.dump(best_model,open('diabetes_model.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
import gradio as gr
import numpy as np
import pickle

model = pickle.load(open('diabetes_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

def predict_diabetes(Pregnancies,Glucose,BloodPressure,SkinThickness,
                     Insulin,BMI,DiabetesPedigreeFunction,Age):

    data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,
                      Insulin,BMI,DiabetesPedigreeFunction,Age]])


    data = scaler.transform(data)

    result = model.predict(data)

    return 'Diabetes' if result[0]==1 else 'Non Diabetes'

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=['number','number','number','number','number','number','number','number'],
    outputs='text',
    title='Diabetes Prediction System'
)

demo.launch(share=True)