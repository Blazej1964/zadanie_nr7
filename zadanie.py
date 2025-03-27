import streamlit as st
import json
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

#CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v3_sport.json'

model_options = {
    'welcome_survey_cluster_names_and_descriptions_v1.json': 'Ogólny',
    'welcome_survey_cluster_names_and_descriptions_v2.json': 'Polska komedia',
    'welcome_survey_cluster_names_and_descriptions_4_horror.json': 'Horror',
    'welcome_survey_cluster_names_and_descriptions_v3_sport.json': 'Polscy sportowcy',
    'welcome_survey_cluster_names_and_descriptions_6_stolica.json': 'Stolice europejskie',
    'welcome_survey_cluster_names_and_descriptions_psychopata.json': 'Psychopata',

}

# Tworzenie rozwijanej listy z opisem modeli
selected_model_key = st.selectbox("Wybierz model opisu osoby z grupy:", list(model_options.values()))

# Znajdź klucz odpowiadający wybranemu opisowi
CLUSTER_NAMES_AND_DESCRIPTIONS = [key for key, value in model_options.items() if value == selected_model_key][0]


######################################################

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

#@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header("Wybierz z poniższej listy:")
    st.markdown("Zaznacz dane, a my pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Przynależysz do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]



########################################


# Nagłówek
st.header(f"Kim zatem są {predicted_cluster_data['name']}")

# Rozwijana lista do wyboru kolon
option = st.selectbox(
    'Wybierz kategorię:',
    ['age','edu_level', 'fav_animals', 'fav_place', 'gender']
)

# Wykres
fig = px.histogram(same_cluster_df, x=option)
fig.update_layout(
    title=f"Rozkład {option} w grupie",
    xaxis_title=option,
    yaxis_title="Liczba osób",
)

st.plotly_chart(fig)

st.metric("Liczba osób w grupie", len(same_cluster_df))
