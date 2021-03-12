from flask import Blueprint, Flask, render_template, make_response, request, send_file
from wtforms import Form, FloatField, validators,StringField, IntegerField
from numpy import exp, cos, linspace
from math import pi, sqrt
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



modsim_api = Blueprint('modsim_api', __name__)

@modsim_api.route('/inventarioEOQ', methods=("POST", "GET"))
def inventario_EOQ():
    class InputForm(Form):
        D = FloatField(
            label='Valor de la demanda anual', default=12000,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Costo de ordenar', default=25.00,
            validators=[validators.InputRequired()])
        CH = FloatField(
            label='Costo de mantenimiento', default=0.50,
            validators=[validators.InputRequired()])
        P = FloatField(
            label='Costo por unidad del producto', default=2.50,
            validators=[validators.InputRequired()])
        TE = IntegerField(
            label='Tiempo de espera del producto en días', default=5,
            validators=[validators.InputRequired()])
        DA = IntegerField(
            label='Días habíles del año', default=250,
            validators=[validators.InputRequired()])
        PE = IntegerField(
            label='Periodo del inventario', default=30,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        D = float(form.D.data)  #Cantidad que se tiene pa distribuir
        Co = float(form.CO.data)
        Ch = float(form.CH.data)
        P = float(form.P.data)
        Tespera = int(form.TE.data)
        DiasAno = int(form.DA.data)
        Periodo = int(form.PE.data)
        Q = round(sqrt(((2*Co*D)/Ch)),2)
        N = round(D / Q,2)
        R = round((D / DiasAno) * Tespera,2)
        T = round(DiasAno / N,2)
        CoT = N * Co
        ChT = round(Q / 2 * Ch,2)
        MOQ = round(CoT + ChT,2)
        CTT = round(P * D + MOQ,2)
        

        # Programa para generar el gráfico de costo mínimo
        indice = ['Q','Costo_ordenar','Costo_Mantenimiento','Costo_total','Diferencia_Costo_Total']
        # Generamos una lista ordenada de valores de Q

        periodo = np.arange(1,Periodo)
        def genera_lista(Q):
            n= Periodo-1
            Q_Lista = []
            i=1
            Qi = Q
            Q_Lista.append(Qi)
            for i in range(1,int(Periodo/2)):
                Qi = Qi - 60
                Q_Lista.append(Qi)

            Qi = Q
            for i in range(int(Periodo/2), n):
                Qi = Qi + 60
                Q_Lista.append(Qi)

            return Q_Lista
        Lista= genera_lista(Q)
        Lista.sort()
        dfQ = pd.DataFrame(index=periodo, columns=indice).fillna(0)
        dfQ['Q'] = Lista
        #dfQ
        for period in periodo:
            dfQ['Costo_ordenar'][period] = D * Co / dfQ['Q'][period]
            dfQ['Costo_Mantenimiento'][period] = dfQ['Q'][period] * Ch / 2
            dfQ['Costo_total'][period] = dfQ['Costo_ordenar'][period] + dfQ['Costo_Mantenimiento'][period]
            dfQ['Diferencia_Costo_Total'][period] = dfQ['Costo_total'][period] - MOQ
        pd.set_option('mode.chained_assignment', None)
        df = dfQ


        dfG = dfQ.loc[:,'Costo_ordenar':'Costo_total']
        dfG
        dfG.plot()

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

       
        return render_template('/metspages/modsim/inventeoq.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, dato1=Q,
        dato2=CoT,dato3=ChT,dato4=MOQ,dato5=CTT,dato6=N,dato7=R,dato8=T)
    else:
        D = None
        CO = None
        CH= None
        P = None
        TE = None
        DA = None
    return render_template('/metspages/modsim/inventeoq.html', form=form, D=D,CO=CO,CH=CH,P=P,TE=TE,DA=DA)


@modsim_api.route('/inventariogeneral', methods=("POST", "GET"))
def inventario_general():
    class InputForm(Form):
        D = IntegerField(
            label='Inventario inicial', default=12000,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Costo de ordenar', default=25.00,
            validators=[validators.InputRequired()])
        P = FloatField(
            label='Costo por unidad del producto', default=2.50,
            validators=[validators.InputRequired()])
        TE = IntegerField(
            label='Cantidad demandada del producto', default=400,
            validators=[validators.InputRequired()])
        DA = IntegerField(
            label='Cantidad que se adquiere de producto al agotarse', default=500,
            validators=[validators.InputRequired()])
        PE = IntegerField(
            label='Tiempo en días que tarda el producto', default=3,
            validators=[validators.InputRequired()])
        CI = IntegerField(
            label='Cantidad a mostrar en la tabla', default=15,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        II = int(form.D.data)  #Cantidad que se tiene pa distribuir
        Co = float(form.CO.data)
        PR = float(form.P.data)
        DP = int(form.TE.data)
        CA = int(form.DA.data)
        TA = int(form.PE.data)
        CI = int(form.CI.data)
        def make_data(product, policy, periods):
            periods += 1
            # Create zero-filled Dataframe
            period_lst = np.arange(periods) # index
            # Abbreviations
            # INV_INICIAL: INV_NETO_INICIALtial inventory position
            # INV_NETO_INICIAL: INV_NETO_INICIALtial net inventory
            # D: Demand
            # INV_FINAL: Final inventory position
            # NETO: Final net inventory
            # LS: Lost sales
            # AVG: Average inventory
            # ORD: order quantity
            # LT: lead time
            header = ['INV_INICIAL','INV_NETO_INICIAL','DEMANDA', 'INV_FINAL',
            'NETO', 'VENTAS', 'INV_PROMEDIO', 'CANT_ORDENAR', 'TIEMPO']
            df = pd.DataFrame(index=period_lst, columns=header).fillna(0)
            # Create a list that will store each period order
            order_l = [Order(quantity=0, lead_time=0)
                for x in range(periods)]
            # Fill DataFrame
            for period in period_lst:
                if period == 0:
                    df['INV_INICIAL'][period] = product.initial_inventory
                    df['INV_NETO_INICIAL'][period] = product.initial_inventory
                    df['INV_FINAL'][period] = product.initial_inventory
                    df['NETO'][period] = product.initial_inventory
                if period >= 1:
                    df['INV_INICIAL'][period] = df['INV_FINAL'][period - 1] + order_l[period - 1].quantity
                    df['INV_NETO_INICIAL'][period] = df['NETO'][period - 1] + pending_order(order_l, period)
                    #demand = int(product.demand())
                    demand = DP
                    # We can't have negative demand
                    if demand > 0:
                        df['DEMANDA'][period] = demand
                    else:
                        df['DEMANDA'][period] = 0
                    # We can't have negative INV_INICIAL
                    if df['INV_INICIAL'][period] - df['DEMANDA'][period] < 0:
                        df['INV_FINAL'][period] = 0
                    else:
                        df['INV_FINAL'][period] = df['INV_INICIAL'][period] - df['DEMANDA'][period]
                    order_l[period].quantity, order_l[period].lead_time = placeorder(product, df['INV_FINAL'][period], policy,
                    period)
                    df['NETO'][period] = df['INV_NETO_INICIAL'][period] - df['DEMANDA'][period]
                    if df['NETO'][period] < 0:
                        df['VENTAS'][period] = abs(df['NETO'][period])
                        df['NETO'][period] = 0
                    else:
                        df['VENTAS'][period] = 0
                        df['INV_PROMEDIO'][period] = (df['INV_NETO_INICIAL'][period] + df['NETO'][period]) / 2.0
                        df['CANT_ORDENAR'][period] = order_l[period].quantity
                        df['TIEMPO'][period] = order_l[period].lead_time

            return df

        def pending_order(order_l, period):
            indices = [i for i, order in enumerate(order_l) if order.quantity]
            sum = 0


            for i in indices:
                if period - (i + order_l[i].lead_time + 1) == 0: 
                    sum += order_l[i].quantity
            return sum

        def demand(self):
            if self.demand_dist == "Constant":
                return self.demand_p1
            

        def lead_time(self):
            if self.leadtime_dist == "Constant":
                return self.leadtime_p1
            

        def __repr__(self):
            return '<Product %r>' % self.name

        def placeorder(product, final_inv_pos, policy, period):
            #lead_time = int(product.lead_time())
            lead_time = 3
            # Qs = if we hit the reorder point s, order Q units
            if policy['method'] == 'Qs' and \
                final_inv_pos <= policy['param2']:
                return policy['param1'], lead_time
            # RS = if we hit the review period R and the reorder point S, order: (S - # final inventory pos)
            elif policy['method'] == 'RS' and \
                period % policy['param1'] == 0 and \
                final_inv_pos <= policy['param2']:
                return policy['param2'] - final_inv_pos, lead_time
            else:
                return 0, 0


        politica = {'method': "Qs",
        'param1': CA,   #Cantidad que se ordena
        'param2': 200
        }

        class Order(object):


            def __init__(self, quantity, lead_time): 
                self.quantity = quantity
                self.lead_time = lead_time

        class product(object):
            def  __init__(self,name,price,order_cost,initial_inventory,demand_dist,demand_p1,
                        demand_p2,demand_p3,leadtime_dist,leadtime_p1,leadtime_p2,leadtime_p3):
                self.name=name
                self.price=price
                self.order_cost=order_cost
                self.initial_inventory=initial_inventory 
                self.demand_dist=demand_dist
                self.demand_p1=demand_p1 
                self.demand_p2=demand_p2 
                self.demand_p3=demand_p3
                self.leadtime_dist=leadtime_dist 
                self.leadtime_p1=leadtime_p1
                self.leadtime_p2=leadtime_p2 
                self.leadtime_p3=leadtime_p3

        producto = product("Mesa",PR,Co,II,"Constant",DP,450,400,"Constant",TA,3,4)

        df = make_data(producto, politica, CI) 
        df

       
        return render_template('/metspages/modsim/inventgen.html', form=form, tables=[df.to_html(classes='data table table-bordered')])
    else:
        D = None
        CO = None
        P = None
        TE = None
        DA = None
        PE = None
        CI = None
    return render_template('/metspages/modsim/inventgen.html', form=form, D=D,CO=CO,P=P,TE=TE,DA=DA,PE=PE,CI=CI)


@modsim_api.route('/inventarioEMPRESA', methods=("POST", "GET"))
def inventario_empresa():
    class InputForm(Form):
        D = IntegerField(
            label='Valor de la demanda anual', default=6720,
            validators=[validators.InputRequired()])
        II = IntegerField(
            label='Inventario inicial', default=5000,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Costo de ordenar', default=15.00,
            validators=[validators.InputRequired()])
        CH = FloatField(
            label='Costo de mantenimiento', default=0.25,
            validators=[validators.InputRequired()])
        P = FloatField(
            label='Costo por unidad del producto', default=3.20,
            validators=[validators.InputRequired()])
        TE = IntegerField(
            label='Tiempo de espera del producto en días', default=1,
            validators=[validators.InputRequired()])
        DA = IntegerField(
            label='Días habíles del año', default=300,
            validators=[validators.InputRequired()])
        PE = IntegerField(
            label='Periodo del inventario', default=30,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        
        D = int(form.D.data)  #Cantidad que se tiene pa distribuir
        II= int(form.II.data)
        Co = float(form.CO.data)
        Ch = float(form.CH.data)
        P = float(form.P.data)
        Tespera = int(form.TE.data)
        DiasAno = int(form.DA.data)
        Periodo = int(form.PE.data)
        Q = round(sqrt(((2*Co*D)/Ch)),2)
        N = round(D / Q,2)
        R = round((D / DiasAno) * Tespera,2)
        T = round(DiasAno / N,2)
        CoT = N * Co
        ChT = round(Q / 2 * Ch,2)
        MOQ = round(CoT + ChT,2)
        CTT = round(P * D + MOQ,2)
        

        # Programa para generar el gráfico de costo mínimo
        indice = ['Q','Costo_ordenar','Costo_Mantenimiento','Costo_total','Diferencia_Costo_Total']
        # Generamos una lista ordenada de valores de Q

        periodo = np.arange(1,Periodo)
        def genera_lista(Q):
            n= Periodo-1
            Q_Lista = []
            i=1
            Qi = Q
            Q_Lista.append(Qi)
            for i in range(1,int(Periodo/2)):
                Qi = Qi - 60
                Q_Lista.append(Qi)

            Qi = Q
            for i in range(int(Periodo/2), n):
                Qi = Qi + 60
                Q_Lista.append(Qi)

            return Q_Lista
        Lista= genera_lista(Q)
        Lista.sort()
        dfQ = pd.DataFrame(index=periodo, columns=indice).fillna(0)
        dfQ['Q'] = Lista
        #dfQ
        for period in periodo:
            dfQ['Costo_ordenar'][period] = D * Co / dfQ['Q'][period]
            dfQ['Costo_Mantenimiento'][period] = dfQ['Q'][period] * Ch / 2
            dfQ['Costo_total'][period] = dfQ['Costo_ordenar'][period] + dfQ['Costo_Mantenimiento'][period]
            dfQ['Diferencia_Costo_Total'][period] = dfQ['Costo_total'][period] - MOQ
        pd.set_option('mode.chained_assignment', None)
        df = dfQ


        dfG = dfQ.loc[:,'Costo_ordenar':'Costo_total']
        dfG
        dfG.plot()

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

        def make_data(product, policy, periods):
            periods += 1
            # Create zero-filled Dataframe
            period_lst = np.arange(periods) # index
            # Abbreviations
            # INV_INICIAL: INV_NETO_INICIALtial inventory position
            # INV_NETO_INICIAL: INV_NETO_INICIALtial net inventory
            # D: Demand
            # INV_FINAL: Final inventory position
            # NETO: Final net inventory
            # LS: Lost sales
            # AVG: Average inventory
            # ORD: order quantity
            # LT: lead time
            header = ['INV_INICIAL','INV_NETO_INICIAL','DEMANDA', 'INV_FINAL',
            'NETO', 'VENTAS', 'INV_PROMEDIO', 'CANT_ORDENAR', 'TIEMPO']
            df = pd.DataFrame(index=period_lst, columns=header).fillna(0)
            # Create a list that will store each period order
            order_l = [Order(quantity=0, lead_time=0)
                for x in range(periods)]
            # Fill DataFrame
            for period in period_lst:
                if period == 0:
                    df['INV_INICIAL'][period] = product.initial_inventory
                    df['INV_NETO_INICIAL'][period] = product.initial_inventory
                    df['INV_FINAL'][period] = product.initial_inventory
                    df['NETO'][period] = product.initial_inventory
                if period >= 1:
                    df['INV_INICIAL'][period] = df['INV_FINAL'][period - 1] + order_l[period - 1].quantity
                    df['INV_NETO_INICIAL'][period] = df['NETO'][period - 1] + pending_order(order_l, period)
                    #demand = int(product.demand())
                    demand = D/12
                    # We can't have negative demand
                    if demand > 0:
                        df['DEMANDA'][period] = demand
                    else:
                        df['DEMANDA'][period] = 0
                    # We can't have negative INV_INICIAL
                    if df['INV_INICIAL'][period] - df['DEMANDA'][period] < 0:
                        df['INV_FINAL'][period] = 0
                    else:
                        df['INV_FINAL'][period] = df['INV_INICIAL'][period] - df['DEMANDA'][period]
                    order_l[period].quantity, order_l[period].lead_time = placeorder(product, df['INV_FINAL'][period], policy,
                    period)
                    df['NETO'][period] = df['INV_NETO_INICIAL'][period] - df['DEMANDA'][period]
                    if df['NETO'][period] < 0:
                        df['VENTAS'][period] = abs(df['NETO'][period])
                        df['NETO'][period] = 0
                    else:
                        df['VENTAS'][period] = 0
                        df['INV_PROMEDIO'][period] = (df['INV_NETO_INICIAL'][period] + df['NETO'][period]) / 2.0
                        df['CANT_ORDENAR'][period] = order_l[period].quantity
                        df['TIEMPO'][period] = order_l[period].lead_time

            return df

        def pending_order(order_l, period):
            indices = [i for i, order in enumerate(order_l) if order.quantity]
            sum = 0


            for i in indices:
                if period - (i + order_l[i].lead_time + 1) == 0: 
                    sum += order_l[i].quantity
            return sum

        def demand(self):
            if self.demand_dist == "Constant":
                return self.demand_p1
            

        def lead_time(self):
            if self.leadtime_dist == "Constant":
                return self.leadtime_p1
            

        def __repr__(self):
            return '<Product %r>' % self.name

        def placeorder(product, final_inv_pos, policy, period):
            #lead_time = int(product.lead_time())
            lead_time = Tespera
            # Qs = if we hit the reorder point s, order Q units
            if policy['method'] == 'Qs' and \
                final_inv_pos <= policy['param2']:
                return policy['param1'], lead_time
            # RS = if we hit the review period R and the reorder point S, order: (S - # final inventory pos)
            elif policy['method'] == 'RS' and \
                period % policy['param1'] == 0 and \
                final_inv_pos <= policy['param2']:
                return policy['param2'] - final_inv_pos, lead_time
            else:
                return 0, 0


        politica = {'method': "Qs",
        'param1': Q,   #Cantidad que se ordena
        'param2': 200
        }

        class Order(object):


            def __init__(self, quantity, lead_time): 
                self.quantity = quantity
                self.lead_time = lead_time

        class product(object):
            def  __init__(self,name,price,order_cost,initial_inventory,demand_dist,demand_p1,
                        demand_p2,demand_p3,leadtime_dist,leadtime_p1,leadtime_p2,leadtime_p3):
                self.name=name
                self.price=price
                self.order_cost=order_cost
                self.initial_inventory=initial_inventory 
                self.demand_dist=demand_dist
                self.demand_p1=demand_p1 
                self.demand_p2=demand_p2 
                self.demand_p3=demand_p3
                self.leadtime_dist=leadtime_dist 
                self.leadtime_p1=leadtime_p1
                self.leadtime_p2=leadtime_p2 
                self.leadtime_p3=leadtime_p3

        producto = product("Mesa",P,Co,II,"Constant",D,450,400,"Constant",T,3,4)

        df2 = make_data(producto, politica, Periodo) 

       
        return render_template('/metspages/modsim/invtpaez.html', form=form, tables=[df.to_html(classes='data table table-bordered')], tables2=[df2.to_html(classes='data table table-bordered')], grafica=plotfile, dato1=Q,
        dato2=CoT,dato3=ChT,dato4=MOQ,dato5=CTT,dato6=N,dato7=R,dato8=T)
    else:
        D = None
        CO = None
        CH= None
        P = None
        TE = None
        DA = None
    return render_template('/metspages/modsim/invtpaez.html', form=form, D=D,CO=CO,CH=CH,P=P,TE=TE,DA=DA)


@modsim_api.route('/lineaespera', methods=("POST", "GET"))
def linea_espera():
    class InputForm(Form):
        D = FloatField(
            label='Escribir valor de landa', default=1.333,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Escribir valor de nu', default=4,
            validators=[validators.InputRequired()])
        CH = IntegerField(
            label='Número de personas en espera', default=10,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        D = float(form.D.data)  
        Co = float(form.CO.data)
        Ch = int(form.CH.data)


        landa = D
        nu = Co
        #La probabilidad de hallar el sistema ocupado o utilización del sistema:
        𝑝=landa/nu
        #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
        𝑃o = 1.0 - (landa/nu)
        #Longitud esperada en cola, promedio de unidades en la línea de espera:
        𝐿𝑞 = landa*landa / (nu * (nu - landa))
        #/ (nu * (nu - landa))
        # Número esperado de clientes en el sistema(cola y servicio) :
        𝐿 = landa /(nu - landa)
        #El tiempo promedio que una unidad pasa en el sistema:
        𝑊 = 1 / (nu - landa)
        #Tiempo de espera en cola:
        𝑊𝑞 = W - (1.0 / nu)
        
        #La probabilidad de que haya n unidades en el sistema:
        n= 1
        𝑃𝑛 = (landa/nu)*𝑛*𝑃o

        dato1= round(landa,3)
        dato2= round(p,3)
        dato3 = round(Po,3)
        dato4 = round(Lq,3)
        dato5 = round(L,1)
        dato6 = round(W,3)
        dato7 = round(Wq,3)
        dato8 = round(Pn,3)


        i = 0
        # Landa y nu ya definidos
        # Atributos del DataFrame
        """
        ALL # ALEATORIO DE LLEGADA DE CLIENTES
        ASE # ALEATORIO DE SERVICIO
        TILL TIEMPO ENTRE LLEGADA
        TISE TIEMPO DE SERVICIO
        TIRLL TIEMPO REAL DE LLEGADA
        TIISE TIEMPO DE INICIO DE SERVICIO
        TIFSE TIEMPO FINAL DE SERVICIO
        TIESP TIEMPO DE ESPERA
        TIESA TIEMPO DE SALIDA
        numClientes NUMERO DE CLIENTES
        dfLE DATAFRAME DE LA LINEA DE ESPERA
        """
        numClientes=Ch
        i = 0
        indice = ['ALL','ASE','TILL','TISE','TIRLL','TIISE','TIFSE','TIESP','TIESA']
        Clientes = np.arange(numClientes)
        dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)
        np.random.seed(100)
        for i in Clientes:
            if i == 0:
                dfLE['ALL'][i] = random.random()
                dfLE['ASE'][i] = random.random()
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i]
                dfLE['TIISE'][i] = dfLE['TIRLL'][i]
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
            else:
                dfLE['ALL'][i] = random.random()
                dfLE['ASE'][i] = random.random()
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
                dfLE['TIISE'][i] = max(dfLE['TIRLL'][i],dfLE['TIFSE'][i-1])
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA","A_SERVICIO","TIE_LLEGADA","TIE_SERVICIO",
        "TIE_EXACTO_LLEGADA","TIE_INI_SERVICIO","TIE_FIN_SERVICIO",
        "TIE_ESPERA","TIE_EN_SISTEMA"])
        dfLE.columns = nuevas_columnas
        df = dfLE
        df2 = dfLE.describe()

        dfLE.plot()

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

       
        return render_template('/metspages/modsim/linespera.html', form=form, tables=[df.to_html(classes='data table table-bordered')], tables2=[df2.to_html(classes='data table table-bordered')], grafica=plotfile, dato1=dato1,
        dato2=dato2,dato3=dato3,dato4=dato4,dato5=dato5,dato6=dato6,dato7=dato7,dato8=dato8)
    else:
        D = None
        CO = None
        CH= None
    return render_template('/metspages/modsim/linespera.html', form=form, D=D,CO=CO,CH=CH)



@modsim_api.route('/lineaesperaaditivo', methods=("POST", "GET"))
def linea_espera_aditivo():
    class InputForm(Form):
        D = FloatField(
            label='Escribir valor de landa', default=1.333,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Escribir valor de nu', default=4,
            validators=[validators.InputRequired()])
        CH = IntegerField(
            label='Número de personas en espera', default=10,
            validators=[validators.InputRequired()])
        M = IntegerField(
            label='Módulo', default=1113,
            validators=[validators.InputRequired()])
        A = IntegerField(
            label='Multiplicador', default=26,
            validators=[validators.InputRequired()])
        X0 = IntegerField(
            label='Semilla', default=109,
            validators=[validators.InputRequired()])
        C = IntegerField(
            label='Incremento', default=111,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        D = float(form.D.data)  
        Co = float(form.CO.data)
        Ch = int(form.CH.data)

        import pandas as pd
        n = Ch*2
        m = int(form.M.data)
        a = int(form.A.data)
        x0 = int(form.X0.data)
        c = int(form.C.data)
        x = [1] * n
        r = [0.1] * n
        for i in range(0, n):
            x[i] = ((a*x0)+c) % m
            x0 = x[i]
            r[i] = x0 / m
        # llenamos nuestro DataFrame
        d = {'Xn': x, 'ri': r}
        dfran = pd.DataFrame(data=d)
        


        landa = D
        nu = Co
        #La probabilidad de hallar el sistema ocupado o utilización del sistema:
        𝑝=landa/nu
        #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
        𝑃o = 1.0 - (landa/nu)
        #Longitud esperada en cola, promedio de unidades en la línea de espera:
        𝐿𝑞 = landa*landa / (nu * (nu - landa))
        #/ (nu * (nu - landa))
        # Número esperado de clientes en el sistema(cola y servicio) :
        𝐿 = landa /(nu - landa)
        #El tiempo promedio que una unidad pasa en el sistema:
        𝑊 = 1 / (nu - landa)
        #Tiempo de espera en cola:
        𝑊𝑞 = W - (1.0 / nu)
        
        #La probabilidad de que haya n unidades en el sistema:
        n= 1
        𝑃𝑛 = (landa/nu)*𝑛*𝑃o

        dato1= round(landa,3)
        dato2= round(p,3)
        dato3 = round(Po,3)
        dato4 = round(Lq,3)
        dato5 = round(L,1)
        dato6 = round(W,3)
        dato7 = round(Wq,3)
        dato8 = round(Pn,3)


        i = 0
        # Landa y nu ya definidos
        # Atributos del DataFrame
        """
        ALL # ALEATORIO DE LLEGADA DE CLIENTES
        ASE # ALEATORIO DE SERVICIO
        TILL TIEMPO ENTRE LLEGADA
        TISE TIEMPO DE SERVICIO
        TIRLL TIEMPO REAL DE LLEGADA
        TIISE TIEMPO DE INICIO DE SERVICIO
        TIFSE TIEMPO FINAL DE SERVICIO
        TIESP TIEMPO DE ESPERA
        TIESA TIEMPO DE SALIDA
        numClientes NUMERO DE CLIENTES
        dfLE DATAFRAME DE LA LINEA DE ESPERA
        """
        numClientes=Ch
        i = 0
        indice = ['ALL','ASE','TILL','TISE','TIRLL','TIISE','TIFSE','TIESP','TIESA']
        Clientes = np.arange(numClientes)
        dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)
        np.random.seed(100)
        for i in Clientes:
            if i == 0:
                dfLE['ALL'][i] = dfran['ri'][i] 
                dfLE['ASE'][i] = dfran['ri'][i+numClientes] 
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i]
                dfLE['TIISE'][i] = dfLE['TIRLL'][i]
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
            else:
                dfLE['ALL'][i] = dfran['ri'][i] 
                dfLE['ASE'][i] = dfran['ri'][i+numClientes] 
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
                dfLE['TIISE'][i] = max(dfLE['TIRLL'][i],dfLE['TIFSE'][i-1])
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA","A_SERVICIO","TIE_LLEGADA","TIE_SERVICIO",
        "TIE_EXACTO_LLEGADA","TIE_INI_SERVICIO","TIE_FIN_SERVICIO",
        "TIE_ESPERA","TIE_EN_SISTEMA"])
        dfLE.columns = nuevas_columnas
        df = dfLE
        df2 = dfLE.describe()

        dfLE.plot()

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

       
        return render_template('/metspages/modsim/linespera.html', form=form, tables=[df.to_html(classes='data table table-bordered')], tables2=[df2.to_html(classes='data  table table-bordered')], grafica=plotfile, dato1=dato1,
        dato2=dato2,dato3=dato3,dato4=dato4,dato5=dato5,dato6=dato6,dato7=dato7,dato8=dato8)
    else:
        D = None
        CO = None
        CH= None
        M = None
        A = None
        X0 = None
        C = None
    return render_template('/metspages/modsim/linespera.html', form=form, D=D,CO=CO,CH=CH,M=M,A=A,X0=X0,C=C)



@modsim_api.route('/lineaesperamultiplicativo', methods=("POST", "GET"))
def linea_espera_multiplicativo():
    class InputForm(Form):
        D = FloatField(
            label='Escribir valor de landa', default=1.333,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Escribir valor de nu', default=4,
            validators=[validators.InputRequired()])
        CH = IntegerField(
            label='Número de personas en espera', default=10,
            validators=[validators.InputRequired()])
        M = IntegerField(
            label='Módulo', default=1113,
            validators=[validators.InputRequired()])
        A = IntegerField(
            label='Multiplicador', default=26,
            validators=[validators.InputRequired()])
        X0 = IntegerField(
            label='Semilla', default=109,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        D = float(form.D.data)  
        Co = float(form.CO.data)
        Ch = int(form.CH.data)

        import pandas as pd
        n = Ch*2
        m = int(form.M.data)
        a = int(form.A.data)
        x0 = int(form.X0.data)
        x = [1] * n
        r = [0.1] * n
        for i in range(0, n):
            x[i] = (a*x0) % m
            x0 = x[i]
            r[i] = x0 / m
        d = {'Xn': x, 'ri': r }
        dfran = pd.DataFrame(data=d)
        


        landa = D
        nu = Co
        #La probabilidad de hallar el sistema ocupado o utilización del sistema:
        𝑝=landa/nu
        #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
        𝑃o = 1.0 - (landa/nu)
        #Longitud esperada en cola, promedio de unidades en la línea de espera:
        𝐿𝑞 = landa*landa / (nu * (nu - landa))
        #/ (nu * (nu - landa))
        # Número esperado de clientes en el sistema(cola y servicio) :
        𝐿 = landa /(nu - landa)
        #El tiempo promedio que una unidad pasa en el sistema:
        𝑊 = 1 / (nu - landa)
        #Tiempo de espera en cola:
        𝑊𝑞 = W - (1.0 / nu)
        
        #La probabilidad de que haya n unidades en el sistema:
        n= 1
        𝑃𝑛 = (landa/nu)*𝑛*𝑃o

        dato1= round(landa,3)
        dato2= round(p,3)
        dato3 = round(Po,3)
        dato4 = round(Lq,3)
        dato5 = round(L,1)
        dato6 = round(W,3)
        dato7 = round(Wq,3)
        dato8 = round(Pn,3)


        i = 0
        # Landa y nu ya definidos
        # Atributos del DataFrame
        """
        ALL # ALEATORIO DE LLEGADA DE CLIENTES
        ASE # ALEATORIO DE SERVICIO
        TILL TIEMPO ENTRE LLEGADA
        TISE TIEMPO DE SERVICIO
        TIRLL TIEMPO REAL DE LLEGADA
        TIISE TIEMPO DE INICIO DE SERVICIO
        TIFSE TIEMPO FINAL DE SERVICIO
        TIESP TIEMPO DE ESPERA
        TIESA TIEMPO DE SALIDA
        numClientes NUMERO DE CLIENTES
        dfLE DATAFRAME DE LA LINEA DE ESPERA
        """
        numClientes=Ch
        i = 0
        indice = ['ALL','ASE','TILL','TISE','TIRLL','TIISE','TIFSE','TIESP','TIESA']
        Clientes = np.arange(numClientes)
        dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)
        np.random.seed(100)
        for i in Clientes:
            if i == 0:
                dfLE['ALL'][i] = dfran['ri'][i] 
                dfLE['ASE'][i] = dfran['ri'][i+numClientes] 
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i]
                dfLE['TIISE'][i] = dfLE['TIRLL'][i]
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
            else:
                dfLE['ALL'][i] = dfran['ri'][i] 
                dfLE['ASE'][i] = dfran['ri'][i+numClientes] 
                dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
                dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
                dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
                dfLE['TIISE'][i] = max(dfLE['TIRLL'][i],dfLE['TIFSE'][i-1])
                dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
                dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
                dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA","A_SERVICIO","TIE_LLEGADA","TIE_SERVICIO",
        "TIE_EXACTO_LLEGADA","TIE_INI_SERVICIO","TIE_FIN_SERVICIO",
        "TIE_ESPERA","TIE_EN_SISTEMA"])
        dfLE.columns = nuevas_columnas
        df = dfLE
        df2 = dfLE.describe()

        dfLE.plot()

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

       
        return render_template('/metspages/modsim/linespera.html', form=form, tables=[df.to_html(classes='data  table table-bordered')], tables2=[df2.to_html(classes='data table table-bordered')], grafica=plotfile, dato1=dato1,
        dato2=dato2,dato3=dato3,dato4=dato4,dato5=dato5,dato6=dato6,dato7=dato7,dato8=dato8)
    else:
        D = None
        CO = None
        CH= None
        M = None
        A = None
        X0 = None

    return render_template('/metspages/modsim/linespera.html', form=form, D=D,CO=CO,CH=CH,M=M,A=A,X0=X0)

