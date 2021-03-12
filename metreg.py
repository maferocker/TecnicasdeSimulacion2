from flask import Blueprint, Flask, render_template, make_response, request, send_file
from wtforms import Form, FloatField, validators,StringField, IntegerField
from numpy import exp, cos, linspace
from math import pi
import io
import random
import os, time, glob
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



metreg_api = Blueprint('metreg_api', __name__)

@metreg_api.route('/reglineal', methods=("POST", "GET"))
def regresion_lineal():
    class InputForm(Form):
        N = StringField(
            label='Escriba los valores de X separados por comas (,)', default='7,1,10,5,4,3,13,10,2',
            validators=[validators.InputRequired()])
        M = StringField(
            label='Escriba los valores de Y separados por comas (,)', default='2,9,2,5,7,11,2,5,14',
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # datos experimentales
        # el DataFrame se llama movil
        prueba = str(form.N.data)
        prueba2 = str(form.M.data)
        ex = prueba.split(",")
        ex2 = prueba2.split(",")
        valX =list(map(float,ex))
        valY =list(map(float,ex2))
        exporta = {'X':valX,
        'Y':valY}

        a = pd.DataFrame(exporta)
        x = a['X']
        y= a['Y']
        df = pd.DataFrame({'X':x,'Y':y})
        x2 = df["X"]**2
        xy = df["X"] * df["Y"]
        df["X^2"] = x2
        df["XY"] = xy
        
        # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
        p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
        p0,p1 = p
        P0 = p0
        P1 = p1
        pfinal = -(p1/p0)
        y_ajuste = p[0]*x + p[1]
        df['Ajuste'] = y_ajuste

        cant = len(df['Y'])

        cant1 = df['X']
        cant2 = df['Y']
        cant3 = df['X^2']
        cant4 = df['XY']

        sum1 = cant1.values.sum()
        sum2 = cant2.values.sum()
        sum3 = cant3.values.sum()
        sum4 = cant4.values.sum()
        # dibujamos los datos experimentales de la recta
        p_datos =plt.plot(x,y,'b.')
        # Dibujamos la recta de ajuste
        p_ajuste = plt.plot(x,y_ajuste, 'r-')
        plt.title('Ajuste lineal por mínimos cuadrados')
        plt.xlabel('Eje x')
        plt.ylabel('Eje y')
        plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper right")
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

       
        return render_template('/metspages/metreg/reglineal.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, cant=cant, sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,P0=P0, P1=P1,fin=pfinal)
    else:
        N = None
        M = None
        cant= None
        sum1 = None
        sum2 = None
        sum3 = None
        sum4 = None
        P0= None
        P1 = None
        fin= None
    return render_template('/metspages/metreg/reglineal.html', form=form, N=N,M=M,cant=cant,sum1=sum1,sum2=sum2,sum3=sum3,sum4=sum4,P0=P0,P1=P1,fin=fin)


@metreg_api.route('/regnolineal', methods=("POST", "GET"))
def regresion_no_lineal():
    class InputForm(Form):
        N = StringField(
            label='Escriba los valores de X separados por comas (,)', default='1850,1860,1870,1880,1890,1900,1910,1920,1930,1940,1950',
            validators=[validators.InputRequired()])
        M = StringField(
            label='Escriba los valores de Y separados por comas (,)', default='23.2,31.4,39.8,50.2,62.9,76.0,92.0,105.7,122.8,131.7,151.1',
            validators=[validators.InputRequired()])
        C = IntegerField(
            label='Dato a predecir', default=1,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # Importar libreria numpy
        # datos experimentales
        # el DataFrame se llama movil
        prueba = str(form.N.data)
        prueba2 = str(form.M.data)
        PRED = int(form.C.data)

        
        ex = prueba.split(",")
        ex2 = prueba2.split(",")
        valX =list(map(float,ex))
        valY =list(map(float,ex2))
        exporta = {'ValTiempo':valX,
        'Y':valY}
        a = pd.DataFrame(exporta)
        cantidad = len(a['ValTiempo'])
        c=2
        x=[0]
        ini=1
        fin=-1
        while c <= cantidad:
            if c % 2 == 0:
                x.append(ini)
                ini=ini+1
                c = c+1
            else:
                x.insert(0,fin)
                fin=fin-1
                c = c+1
        a['X'] = x
        x = a['X']
        y= a['Y']
        ValTiempo = a["ValTiempo"]
        df = pd.DataFrame({'ValTiempo':ValTiempo,'X':x,'Y':y})
        x2 = df["X"]**2
        x3 = df["X"]**3
        x4 = df["X"]**4
        xy = df["X"] * df["Y"]
        x2y = x2 * df["Y"]
        df["X^2"] = x2
        df["X^3"] = x3
        df['X^4'] = x4
        df["XY"] = xy
        df["X^2Y"] = x2y

        cant1 = df['X']
        cant2 = df['Y']
        cant3 = df['X^2']
        cant4 = df['X^3']
        cant5 = df['X^4']
        cant6 = df['XY']
        cant7 = df['X^2Y']

        sum1 = cant1.values.sum()
        sum2 = cant2.values.sum()
        sum3 = cant3.values.sum()
        sum4 = cant4.values.sum()
        sum5 = cant5.values.sum()
        sum6 = cant6.values.sum()
        sum7 = cant7.values.sum()


        p = np.polyfit(x,y,2)
        p0,p1,p2 = p
        P0 = p0
        P1 = p1
        P2 = p2
        #print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
        y_ajuste = p[0]*x*x + p[1]*x + p[2]
        n=x.size
        x1 = []
        x2 = []
        for i in [PRED]:
            y1_ajuste = p[0]*i*i + p[1]*i + p[2]
            x1.append(i)
            x2.append(y1_ajuste)
        df["Ajuste"]=y_ajuste
        dp = pd.DataFrame({'ValTiempo':'Dato buscado','X':PRED, 'Y':[0],'Ajuste':x2})
        res=x2[-1]
        df = df.append(dp,ignore_index=True)
        
        p_datos =plt.plot(x,y,'b.')
        # Dibujamos la curva de ajuste
        p_ajuste = plt.plot(x,y_ajuste, 'r-')
        plt.title('Ajuste Polinomial por mínimos cuadrados')
        plt.xlabel('Eje x')
        plt.ylabel('Eje y')
        plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

        
        return render_template('/metspages/metreg/regnolineal.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,sum5=sum5,sum6=sum6,sum7=sum7,P0=P0, P1=P1,P2=P2,cant=cantidad,pron=PRED,res=res)
    else:
        N = None
        M = None
        C = None
        sum1 = None
        sum2 = None
        sum3 = None
        sum4 = None
        sum5 = None
        sum6 = None
        sum7= None
        P0= None
        P1 = None
        P2= None
        cantidad = None
        PRED= None
        res= None
    return render_template('/metspages/metreg/regnolineal.html', form=form, N=N, M=M, C=C,sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,sum5=sum5,sum6=sum6,sum7=sum7,P0=P0, P1=P1,P2=P2,cant=cantidad,pron=PRED,res=res)




