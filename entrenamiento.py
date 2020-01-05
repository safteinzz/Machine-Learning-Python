# -*- coding: utf-8 -*-
"""
Entrenamiento vuelos
Autores: Pablo, Roberto y Sergio
"""
### LIBRERIAS ###

# => imports principales
import sys
import numpy as np
import pandas as pd

# => import clase local
from pandasmodel import PandasModel

# => entorno gráfico
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic

# => para alerts
import ctypes  # An included library with Python install.

# => algoritmos
## partidor datos Train VS test
from sklearn.model_selection import train_test_split
## Regresion Logistica
from sklearn.linear_model import LogisticRegression
## Support Vector Machines
from sklearn.svm import SVC
## KNN - K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

### Variables Globales ###
df_entrenar_global = None

### CUERPO DEL PROGRAMA ###

# => DIALOGO DEL PROGRAMA a usar diseñado en QT designer
Ui_MainWindow, QtBaseClass = uic.loadUiType("entrenamiento.ui")

# => Clase del Dialogo ppal.
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)        
        
        # Cargar datos excel
        self.ui.excel_cargar_button.clicked.connect(self.Cargadatos_Entrenar)
        
        # Lanzar entrenamiento según radiobutton
        self.ui.entrenar_button.clicked.connect(self.Lanzar_Entrenamiento)        

    ##
    # Carga del dataset de entrenamiento => dataframe
    ##
    def Cargadatos_Entrenar( self ):
        
        """
        #dialogo selector de fichero        
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        """
        fileName='dataset_TRAIN.xls'
        
        if not fileName:
            Messagebox('No existe el fichero de carga '+fileName, 'Cargadatos_Entrenar', 1)        
            return        
        
        self.ui.fichEntrenamiento_txt.setText(fileName)            
        
        df_entrenar = pd.read_excel (fileName) #cargamos en dataframe
        
        ############## consola
        
        # * mostramos datos cargados al dataframe
        print ("=================================")
        print ("Datos cargados")            
        print ("=================================")
        
        # muestra de registros
        print ("=================================")
        print ("Muestra de registros")            
        print ("=================================")
        print (df_entrenar.head())            
        
        # estructura de datos                        
        print ("=================================")
        print ("nº filas, columnas")
        print ("=================================")                        
        print (df_entrenar.shape)            
        
        # verificamos tipos de datos
        print ("=================================")
        print ("verificamos tipos de datos")
        print ("=================================")            
        print (df_entrenar.info())
        
        # comprobamos campos con datos nulos/erroneos
        print ("===========================================")
        print ("comprobamos campos con datos nulos/erroneos")
        print ("===========================================")            
        print (pd.isnull(df_entrenar).sum())            
        
        # estadísticas dataset
        print ("===========================================")
        print ("estadísticas dataset")
        print ("===========================================")            
        print (df_entrenar.describe())            
        
        # transformos campo label de texto a numerico            
        df_entrenar['salida_retrasada'].replace(['Y', 'N'], [1,0], inplace=True)
        
        # eliminamos columna "nada" que no aporta al entrenamiento
        df_entrenar.drop(['nada'], axis=1, inplace=True)                                  
                                
        ############## dialogo
        
        # convertimos en modelo con clase PandasModel
        model_entrenar = PandasModel(df_entrenar) 
        
        # visualizar en ui.tableView
        self.ui.mitableView.setModel(model_entrenar)
        
        # datos estadisticos dataframe
        dt_entrenar_info_string=""
        dt_entrenar_info_string+=" - Nº registros: " + str(df_entrenar.shape[0]) ## Gives no. of rows/records 
        dt_entrenar_info_string+="\n"
        dt_entrenar_info_string+=" - Nº columnas: " + str(df_entrenar.shape[1]) ## Gives no. of columns 
        dt_entrenar_info_string+="\n\n"
        dt_entrenar_info_string+="- Más info a poner"
        
        self.ui.label_Info.setText(dt_entrenar_info_string)
            
        # indicamos que es la variable global
        global df_entrenar_global
        # asignamos a la variable global
        df_entrenar_global = df_entrenar            
            
    ##
    # Lanzar entrenamiento
    ##
    def Lanzar_Entrenamiento(self):        
        
        # chequeamos que hemos seleccionado una opción => valorar combobox        
        if not self.ui.rd_Algor_RL.isChecked() and not self.ui.rd_Algor_SVM.isChecked() and not self.ui.rd_Algor_KNN.isChecked():                    
            Messagebox('No ha seleccionado ningún algoritmo', 'Lanzar_Entrenamiento', 1)
            return
                                
        ### Entrenamiento        
        global df_entrenar_global # indicamos que es la variable global                        
        # igualamos a la variable global
        df_entrenar =  df_entrenar_global        
        
        # Separamos columna con la info de salida_retrasada                        
        X =  np.array(df_entrenar.drop(['salida_retrasada'], 1)) # variables del modelo        
        Y =  np.array(df_entrenar['salida_retrasada']) # resultado
        
        # separamos datos en entrenamiento y prueba para testear algoritmos
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)
        
        Y_pred=0
        str_algoritmo = ""        
        
        print('\nEntrenamiento lanzado')
        
        
        if self.ui.rd_Algor_RL.isChecked():
            
            # Regresion logistica
            logreg = LogisticRegression()
            logreg.fit(X_Train, Y_Train)
            Y_pred = logreg.predict(X_Test)            
            
            print('Precisión Regresión logística:')
            
            print(logreg.score(X_Train, Y_Train))            
            str_algoritmo="Logístic Regression"        
            
        elif self.ui.rd_Algor_SVM.isChecked():            
            # Support Vectors Machines
            svc = SVC()
            svc.fit(X_Train, Y_Train)
            Y_pred = svc.predict(X_Test)
            
            print('Precisión SVM')
            
            print(svc.score(X_Train, Y_Train))            
            str_algoritmo="SVM - Support Vectors Machine"
            
        elif self.ui.rd_Algor_KNN.isChecked():            
            # KNN - K Nearest Neighbors
            knn = KNeighborsClassifier(n_neighbors = 3)
            knn.fit(X_Train, Y_Train)
            Y_pred = knn.predict(X_Test)
            
            print('Precisión KNN')
            
            print(knn.score(X_Train, Y_Train))            
            str_algoritmo="KNN - Logístic Regression"
        
        print(Y_pred)        
        
        print('\n\nEntrenamiento con ' + str_algoritmo + ' finalizado')        
        
            
### OTRAS FUNCIONES

##
## función Messagebox para alertas
##
def Messagebox(text, title, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)            
            
##
## Función MAIN
##
def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()