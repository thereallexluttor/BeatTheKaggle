import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import dataframe_cleaner as dc
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import dataframe_info as di
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import matplotlib.pyplot as plt


def model_analyser(final_df, columns):
    datax = {"A": [], "B": [], "C": []}
    data = final_df
    max_score = 0
    for i in range(len(columns)):
        X = data.drop(columns[i], axis=1)
        y = data[columns[i]]
    

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        # Instanciar el modelo LazyClassifier
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

        # Entrenar el modelo y obtener los resultados
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        for index, row in models.iterrows():
            #st.markdown('Modelo:')
            Accu = row.loc['Accuracy']
            #index
            #st.markdown('Accuracy:')
            #Accu
            vari = columns[i]
            #vari
            if Accu != 1:
              datax["A"].append(index)
              datax["B"].append(Accu)
              datax["C"].append(vari)


    
    daframef = pd.DataFrame(datax)
    return models, predictions, daframef


st.title('Machine Learning Automatizado')
st.markdown('La magia de la IA a velocidades disruptivas.')

# load the file to do the real f*cking magic!
data_file = st.file_uploader("Subir archivo .csv !")

# Catch the file and load it!
if data_file is not None:
    data_df = pd.read_csv(data_file)

    # instance the cleanes
    cleaner = dc.CleanDataframe(data_df)

    # fix the names
    cleaner.name_formater()
    
    # clean the data
    data_df_final = cleaner.clean_dataframe()

    # instante the categorical cleanes
    df_no_categorical = di.InfoDataframe(data_df_final)

    # get the next df
    final_df = df_no_categorical.remove_categorical_columns()

    final_df

    # Crear una lista con el nombre de las variables
    columnas = final_df.columns.tolist()
        
    
    
    #analizar todos los modelos para todas las variables
    models, predictions,metadf = model_analyser(final_df,columnas)
    #metadf     
    max_row = metadf.loc[metadf[['B']].idxmax()]
    max_row

         
    # Obtener el modelo con el mejor rendimiento
    #best_model = max(models.items(), key=lambda x: x[1]['Accuracy'])

    # Imprimir el modelo con el mejor rendimiento
    #best_model



    
    # crear un diccionario de modelos y predicciones
    model_dict = dict(zip(models, predictions))
    # plotear los resultados
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(model_dict)), list(model_dict.values()), align='center')
    ax.set_yticks(range(len(model_dict)))
    ax.set_yticklabels(list(model_dict.keys()))
    ax.set_xlabel('Accuracy')
    ax.set_title('LazyClassifier Model Comparison')
    st.pyplot(fig)
    
