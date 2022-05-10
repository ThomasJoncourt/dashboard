import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import requests
import plotly.express as px
import matplotlib.pyplot as plt

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

st.title('Prix médian des maisons en Californie par quartiers')

df = pd.DataFrame(X,columns=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'])
df["Prix median en $"] = y *100000

token = "pk.eyJ1IjoidGhvbWFzamN0IiwiYSI6ImNsMnN6NnVkYzAzbDczaWxya28xMTI5aW0ifQ.zFr4fMzbO2iMmBeAydU4Mg"

fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',color="Prix median en $",
                        color_discrete_sequence=["blue"], zoom=5,center=dict(lat=36.95, lon=-120))
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)

st.plotly_chart(fig, use_container_width=True)

st.title('Estimation prix médian :')

with st.form("my_form"):

    st.write("Vueillez remplir les champs du formulaire !")

    MedInc = st.number_input('Insert : Median income in block group in k$',min_value=0.0)
    st.write('Median income in block group is ', MedInc)

    HouseAge = st.number_input('Insert : Median house age in block group',min_value=0.0)
    st.write('Median house age in block group is ', HouseAge)

    AveRooms = st.number_input('Insert : Average number of rooms per household',min_value=0.0)
    st.write('Average number of rooms per household is ', AveRooms)

    AveBedrms = st.number_input('Insert : Average number of bedrooms per household',min_value=0.0)
    st.write('Average number of bedrooms per household is ', AveBedrms)

    Population = st.number_input('Insert : Block group population',min_value=0.0)
    st.write('Block group population is ', Population)

    AveOccup = st.number_input('Insert : Average number of household members',min_value=0.0)
    st.write('Average number of household members is ', AveOccup)

    Latitude = st.number_input('Insert : Block group latitude',min_value=30.0,max_value=45.0)
    st.write('Block group latitude is ', Latitude)

    Longitude = st.number_input('Insert : Block group longitude',min_value=-130.0,max_value=-110.0)
    st.write('Block group longitude is ', Longitude)

    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Félicitation vous avez bien rempli le formulaire")
        col=["MedInc", "HouseAge" ,"AveRooms","AveBedrms","Population" ,"AveOccup" ,"Latitude","Longitude"]
        X_te=pd.DataFrame([(MedInc, HouseAge ,AveRooms,AveBedrms,Population ,AveOccup ,Latitude,Longitude)], columns=col )
        headers = {'Content-Type': 'application/json'}
        url = 'http://127.0.0.1:5000/invocations'
        data = X_te.to_json(orient='split')
        response = requests.post(url,headers=headers, data=data)
        responseJson = response.json()
        pred = round(responseJson[0]*100000,0)
        st.write("Price Prédiction : ",pred," $")
        f = plt.figure()
        plt.hist(x=df["Prix median en $"])
        plt.plot([pred,pred],[0,5000],color="red")
        plt.show()
        st.plotly_chart(f, use_container_width=True)

# streamlit run dashboard.py







