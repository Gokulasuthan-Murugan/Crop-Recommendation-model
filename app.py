import streamlit as st
import pickle


pickle_model=open("soil_classification.pkl",'rb')
classifier=pickle.load(pickle_model)

def prediction(N,P,K,Temperature,Humidity,ph,Rainfall):
    prediction=classifier.predict([[N,P,K,Temperature,Humidity,ph,Rainfall]])
    print(prediction)
    return prediction


def main():
    st.title('Crop Recommendation')
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
       background-image: url("https://media.sciencephoto.com/image/c0236973/800wm/C0236973-Ripe_cotton_crop.jpg");

       background-size: cover;
    }
    </style>
    """

    st.markdown(page_element, unsafe_allow_html=True)
    N=st.text_input("N","Type Here")
    P=st.text_input("P","Type Here")
    K=st.text_input("K","Type Here")
    Temperature=st.text_input("Temperature","Type Here")
    Humidity=st.text_input("Humidity","Type Here")
    ph=st.text_input("ph","Type Here")
    Rainfall=st.text_input("Rainfall","Type Here")
    result=""

    if st.button("Predict crop"):
        result=prediction(N,P,K,Temperature,Humidity,ph,Rainfall)
    st.success(f'The Recommended crop is {result}')
    
    st.toast(f'The Recommended crop is {result}')
    if st.button("About"):
        st.text("Data Driven Crop Recommendation ")
        st.text("It takes the Value of nitrogen,phosperus,pottasium of the soil with temperature, humidity,rainfall and return crop suitable for this inputs")


main()

