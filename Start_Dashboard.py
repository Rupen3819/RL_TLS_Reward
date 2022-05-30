print('running')
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
pd.options.display.float_format = '{:.2f}'.format
from datetime import datetime
import plotly.figure_factory as ff
import dash_table
from statistics import mean
import os
import plotly.io as pio
import webbrowser
from threading import Timer
import plotly.graph_objects as go
import random
from Dashboard_Backend.Store_Data import *
from Dashboard_Backend.add_TLS import *
from pathlib import Path

dec=2
required_clicks=1




external_stylesheets = [Path('assets/style.css')]

app = dash.Dash(__name__,title='UrbanAI',external_stylesheets=external_stylesheets)

#auth = dash_auth.BasicAuth(
 #   app,
#    VALID_USERNAME_PASSWORD_PAIRS
#)

all_comparison = {
    'reward': [''],
    'waiting time':[''],
    'waiting time hist':[''],
    'emission':[''],
    'q_values':[''],
    'actions':[''],
    'reward_replay':[''],
 #   'actions hist':['']
}

scenarios={'5-6_(1380)', '8-9_(2600)', '17-18_(3100)', '23-24_(470)'}
scenarios_list=['5-6_(1380)', '8-9_(2600)', '17-18_(3100)', '23-24_(470)']




app.layout = html.Div(style={'backgroundColor': '#005BA2','padding-bottom':0, 'margin':'0px', 'padding':'15px', 'padding-bottom':'500px'},
    children=[html.Div(className='image', style={'backgroundColor': '#E7E8E8','margin':0,'padding-bottom': 0, 'height':60},
                      children=[
                          html.Img(
                        src=app.get_asset_url('THI.jpg'),
                          style={'padding-top': '10px','padding-left':0,'margin':0,'overflow': 'hidden','width':'10%',
                                 'display': 'inline-block','padding-top':0}),
                          html.P(children='AIMotion - RL for TLS',
                                       style={'color':'black', 'padding-left':20,'font-weight': 'bold','display': 'inline-block',
                                              'font-size':'25px','vertical-align':'top'}),
                      ])
                 ,
        html.Div(className='row',style={'backgroundColor': '#E7E8E8','font-size':'16px'},
                 children=[
                    html.Div(className='four columns div-user-controls', style={'backgroundColor': '#E7E8E8','padding-top':'35px','padding-left':20},
                             children=[
                                 html.P(children='SELECT TRAFFIC SCENARIO',
                                       style={'color':'black','font-weight': 'bold'}),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='traffic', options=[{'label': i, 'value': i} for i in scenarios],
                                                      multi=False, value=scenarios_list[0],
                                                      style={'backgroundColor': '#FFFFFF', 'margin-left':0},
                                                      className='run_selector'
                                                      ),]),                                 
                                 html.P(children='SELECT SETUP FOR ANALYSIS',
                                       style={'color':'black','font-weight': 'bold','padding-top':'20px'}),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='run', options=[{'label': i, 'value': i} for i in Setups],
                                                      multi=True, value=Setups[0],
                                                      style={'backgroundColor': '#FFFFFF', 'margin-left':0},
                                                      className='run_selector'
                                                      ),
                                     ],
                                     ),
                                 html.P(children='SELECT DATA STREAM',
                                       style={'color':'black','padding-top':25,'font-weight': 'bold'}),
                                 dcc.Dropdown(
                                        id='pre-comparison',
                                        options=[{'label': k, 'value': k} for k in all_comparison.keys()],
                                        value='reward',
                                        multi=False,style={'backgroundColor': '#FFFFFF', 'margin-left':0,'padding-left':0}),
#                                 html.P(children='Replay the Traffic',
 #                                      style={'color':'black','padding-top':25,'font-weight': 'bold'}),
                                 dcc.Graph(id='graph_replay',style={'backgroundColor': '#E7E8E8','fontColor':'black', 'padding-bottom':'20px'},),
                                 dcc.Slider(id='slider',min=0,max=Dataframes_replay[0][2]['step'].max(), value=5, marks={0:'0s',Dataframes_replay[0][2]['step'].max():'s'}),
                                 html.Div(id='slider-output', style={'color':'black','padding-left': '50'})
##                              
                                 ##dcc.Dropdown(
                                 
##                                    id='comparison',
##                                    value='p-rate per worker (1/(min/worker))',
##                                    style={'backgroundColor': '#FFFFFF', 'margin-left':5,'padding-left':-10,'color':'#010A33'}),
##                                 html.P(children='DOWNLOAD ANALYSIS?',
##                                       style={'color':'black','padding-top':75,'font-weight': 'bold'}),
##                                 dcc.Dropdown(
##                                 id='download',
##                                 options=[{'label':'YES', 'value':'YES'},
##                                        {'label':'NO', 'value':'NO'}],
##                                 style={'backgroundColor': '#FFFFFF', 'margin-left':0, 'margin-right':150},
##                                 value='NO')
##                                 ,
                                 #html.Button('Download', id='download', n_clicks=0, style={'backgroundColor': '#FFFFFF', 'margin-top':50, 'margin-left':5}),
                                 #html.P(id='placeholder',style={'color':'black', 'padding-left':5,'padding-top':10}),
                                 #html.P(id='placeholder2',style={'color':'black', 'padding-left':5,'padding-top':10})
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',style={'backgroundColor': '#E7E8E8','fontColor':'black', 'padding-right':'40px','padding-top':'30px'},
                             children=[
                                 dcc.Graph(id='graph', style={'backgroundColor': '#E7E8E8','fontColor':'black', 'padding-bottom':'20px'},),
                                 
                                 dash_table.DataTable(id='data-table',
                                                   columns=[{"name": i, "id": i, 'format': {"specifier": ".2f"}} for i in Dataframes_average.columns],
                                                    sort_action="native",
                                                    data=Dataframes_average.to_dict('records'),
                                                      #style_header={'backgroundColor': '#E7E8E8'},
                                                                                    style_cell={
                                                                                        #'backgroundColor': '#E7E8E8',
                                                                                        'color': 'black',
                                                                                        'font-family':'IBM Plex Sans',
                                                                                        'height': 'auto',
                                                                                        'whiteSpace': 'normal',
                                                                                        'textAlign': 'left'},
                                                                                    style_table={
                                                                                        'height': '200px',
                                                                                        'overflowY': 'auto',
                                                                                        'color':'black'
                                                                                        
                                                                                    },
                                                )
##
                             ])
                              ]),
              html.Div(className='image', style={'backgroundColor': '#005BA2','margin':0,'padding-bottom': 500,' text-align': 'center'},
                      children=[
##                          html.Img(
##                        src=app.get_asset_url('arculus_new.png'),
##                          style={'padding-top': 40, 'padding-left':'calc(100% / 2.1)','max-height':'150px','max-width':'150px','overflow': 'hidden',
##                                 'display': 'inline-block',  'margin-left':'auto',
##                                  'margin-right':'auto', 'text-align': 'center'}),
                      ])
              
        ]



)
@app.callback(
    Output('slider-output', 'children'),
    Input('slider', 'value'))

def update_slider_output(slider):
    return 'Currently in second "{}"'.format(slider)

@app.callback(
    Output('slider', 'max'),
    Input('run', 'value'))

def update_slider(run):
    if type(run)==str:
        run_list=list()
        run_list.append(run)
        run=run_list

    
    for b in range(len(Setups)):
        if run[0]==Setups[b]:
            return Dataframes_replay[b]['step'].max()

@app.callback(
    Output('run', 'multi'),
    Input('pre-comparison', 'value'))
def update_multi(selection):
    if selection=='q_values':
        return False
    else:
        return True 
    

@app.callback(
    Output('graph_replay', 'figure'),
    Input('run', 'value'),
    Input('slider','value'),
    Input('traffic', 'value'))
def update_replay(run, slider,traffic):
    for s, scenario in enumerate(scenarios):
        if traffic==scenario:
            current_traffic=s

            
    if type(run)==str:
        run_list=list()
        run_list.append(run)
        run=run_list

    else:
        run_list=run


    for s, scenario in enumerate(scenarios):
        if traffic==scenario:
            current_traffic=s

    for b in range(len(Setups)):
        if run[0]==Setups[b]:
            #print(run[i])
            df_position=Dataframes_replay[b][current_traffic]
            df_action=Dataframes_actions[b]
            label=Setups[b]
    df_position=df_position[df_position['step'] == slider]        
    fig = px.scatter(df_position, x="x-position", y="y-position",width=500, height=500)
    print(df_action)
    print(current_traffic)
    print(type(current_traffic))
    print(df_action[current_traffic])
    current_action=df_action[current_traffic].loc[df_action[current_traffic]['step']==slider].values.tolist()
    current_action=int(current_action[0][0])
    print(current_action)
    #current_action=0
    fig=add_TLS(current_action,fig)

    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_layout(yaxis_showticklabels=False, yaxis_visible=False,)
    fig.update_layout(xaxis_showticklabels=False,xaxis_visible=False,)

    fig.update_xaxes(range=[-100, 100])
    fig.update_yaxes(range=[-100, 100])



    return fig


@app.callback(
    Output('graph', 'figure'),
    Input('run', 'value'),
    Input('slider','value'),
    Input('pre-comparison', 'value'),
    Input('traffic','value'))
def update_graph(run, slider, analysis,traffic):
    if type(run)==str:
        run_list=list()
        run_list.append(run)
        run=run_list

    else:
        run_list=run

    for s, scenario in enumerate(scenarios):
        if traffic==scenario:
            current_traffic=s

        
    if analysis=='reward':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_reward[b])
                    label_list.append(Setups[b])


        fig = go.Figure()

        print(df_list[0])

        for i in range(len(df_list)):
            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i]['reward'],
                    mode='lines+markers',
                    name=label_list[i],
                    ))

        fig.update_yaxes(type="linear")
        fig.update_layout(title='Reward During Training')
        
        
        
        
        return fig





    if analysis=='waiting time':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_waiting[b][current_traffic])
                    label_list.append(Setups[b])


        fig = go.Figure()


        for i in range(len(df_list)):
            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i]['waiting time'],
                    mode='lines+markers',
                    name=label_list[i],
                    ))
        fig.add_vline(x=slider, line_width=0.5, line_dash="dash", line_color="black")
        
        fig.update_yaxes(type="linear")
        fig.update_layout(title='Waiting Time')
        
        
        
        
        return fig

    if analysis=='q_values':
        if type(run)==str:
            run_list=list()
            run_list.append(run)
            run=run_list
        for b in range(len(Setups)):
            if run[0]==Setups[b]:
                    #print(run[i])
                df=Dataframes_qvalues[b][current_traffic]
                label=Setups[b]
        fig=go.Figure()
        print(df)
        fig.add_trace(go.Scatter(x=df.index, y=df['action 1'],
                    mode='lines+markers',name='PHASE_NS_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 2'],
                    mode='lines+markers',name='PHASE_NSL_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 3'],
                    mode='lines+markers',name='PHASE_EW_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 4'],
                    mode='lines+markers',name='PHASE_EWL_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 5'],
                    mode='lines+markers',name='PHASE_N_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 6'],
                    mode='lines+markers',name='PHASE_W_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 7'],
                    mode='lines+markers',name='PHASE_S_GREEN'))
        fig.add_trace(go.Scatter(x=df.index, y=df['action 8'],
                    mode='lines+markers',name='PHASE_E_GREEN'))
        fig.add_vline(x=slider, line_width=0.5, line_dash="dash", line_color="black")


        return fig

    if analysis=='emission':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_emission[b][current_traffic])
                    label_list.append(Setups[b])


        fig = go.Figure()


        for i in range(len(df_list)):
            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i]['emission'],
                    mode='lines+markers',
                    name=label_list[i],
                    ))
        fig.add_vline(x=slider, line_width=0.5, line_dash="dash", line_color="black")
        
        fig.update_yaxes(type="linear")
        fig.update_layout(title='Waiting Time')

        return fig


    if analysis=='waiting time hist':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_waiting_car[b][current_traffic])
                    label_list.append(Setups[b])


        fig = go.Figure()


        for i in range(len(df_list)):
            fig.add_trace(go.Histogram(x=df_list[i]['waiting time car'], nbinsx=10, name=label_list[i]))
        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)

        return fig

    if analysis=='actions':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_actions[b][current_traffic])
                    label_list.append(Setups[b])

        fig=go.Figure()

        for i in range(len(df_list)):
            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i]['action'], mode='lines+markers', name=label_list[i]))

        fig.update_yaxes(type="category")
        fig.add_vline(x=slider, line_width=0.5, line_dash="dash", line_color="black")

        return fig

    if analysis=='reward_replay':
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Setups)):
                if run[i]==Setups[b]:
                    #print(run[i])
                    df_list.append(Dataframes_reward_replay[b][current_traffic])
                    label_list.append(Setups[b])

        fig=go.Figure()

        for i in range(len(df_list)):
            fig.add_trace(go.Scatter(x=df_list[i]['step'], y=df_list[i]['reward'], mode='lines+markers', name=label_list[i]))

        fig.add_vline(x=slider, line_width=0.5, line_dash="dash", line_color="black")

        return fig        

    
    

    
    

##@app.callback(
##    Output('data-table','data'),
##    Input('run','value'))
##def update_table(run):
##    if type(run)==str:
##        run_list=list()
##        run_list.append(run)
##        run=run_list
##    else:
##        run_list=run
##
##    run=run_list[0]
##    for i in range(len(Runs)):
##            if run==Runs[i]:
##                current=i
##                
##    data=Dataframes_edited[current].to_dict('records')
##
##    return data

##@app.callback(
##    Output('graph', 'figure'),
##    Input('run', 'value'),
##    Input('pre-comparison', 'value'))
##def update_graph(run,comp):
##    if type(run)==str:
##        run_list=list()
##        run_list.append(run)
##        run=run_list
##
##    else:
##        run_list=run 
##    label_list=list()
##    df_list=list()
##    hover_list=list()
##    #print(run_list)
##
##    ###Utilization visualization
##    if comp=='utilization':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    #print(run[i])
##                    df_list.append(Dataframes_ut[b])
##                    label_list.append(Runs[b])
##        fig = go.Figure()
##
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i]['x-values'], y=df_list[i]['means'],mode='lines',name=label_list[i]))
##
##        fig.update_layout(
##            xaxis_title='simtime',
##            yaxis_title='utilization [%]',
##            title='Average AGV Utilization',
##        font_family='IBM Plex Sans')
##        #print('2')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        #print('3')
##        
##        return fig
##
##    ###Process time visualization
##    if comp=='anfahrt':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_anfahrt[b])
##                    hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig=go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i],mode='lines',name=label_list[i],text=hover_list[i]['Name FmJob']))
##        fig.update_layout(
##                          
##            xaxis_title='job index',
##            yaxis_title=comp,
##        font_family='IBM Plex Sans')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        return fig
##
##
##    if comp=='processing time':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_process[b])
##                    hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig=go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i],mode='lines',name=label_list[i],text=hover_list[i]['Name FmJob']))
##        fig.update_layout(
##                          
##            xaxis_title='job index',
##            yaxis_title=comp,
##        font_family='IBM Plex Sans')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        return fig
##
##    if comp=='duration till arrival target 1':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_target1[b])
##                    hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig=go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i],mode='lines',name=label_list[i],text=hover_list[i]['Name FmJob']))
##        fig.update_layout(
##                          
##            xaxis_title='job index',
##            yaxis_title=comp,
##        font_family='IBM Plex Sans')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        return fig
##
##    if comp=='schnelldreher':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_schnelldreher[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig=go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i][0].index, y=df_list[i][0],mode='lines',name=label_list[i]))
##        fig.update_layout(
##                          
##            xaxis_title='job index',
##            yaxis_title=comp,
##        font_family='IBM Plex Sans')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        return fig
##
##    if comp=='leadtime difference':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_leadtime[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig=go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(x=df_list[i].index, y=df_list[i],mode='lines',name=label_list[i]))
##        fig.update_layout(
##                          
##            xaxis_title='job index',
##            yaxis_title=comp,
##        font_family='IBM Plex Sans')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="white")
##        return fig
##
##
##    if comp=='leadtime difference analysis':
##        colors=['#68BABB','#605A79','#000000','#E7E8E8','#6DACC0']
##        df_analysis_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    Dataframes_management[b]['title+']=Dataframes_management[b]['FMJob title']#+' '+Dataframes[b]['Target 2']
##                    Dataframes_management[b]['hover']=Dataframes_management[b]['Target 2']
##                    Dataframes_leadtime=Dataframes_management[b][Dataframes_management[b]['Lead time deviation [s]']<0]
##                    print(Dataframes_leadtime)
##                    Dataframes_leadtime=Dataframes_leadtime[(Dataframes_leadtime['FMJob title'].str.contains("LG") == False)]
##                    df_list.append(Dataframes_leadtime)
##                    #df_analysis_list.append(Dataframes[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        if len(label_list)>0:
##            print(df_list[0]['Target 2'])
##            data=[
##                go.Bar(
##                name='Run 1',
##                x=df_list[0]['title+'],
##                y=df_list[0]['Duration FMJob assigned to AGV [s]'],
##                hovertext=df_list[0]['hover'],
##                offsetgroup=0,
##                #base=jis_df["sim time (min)"],
##                orientation='v',
##                marker_color=colors[0],
##                showlegend=False),
##                
##                go.Bar(
##                name='Assigned to AGV',
##                x=df_list[0]['title+'],
##                y=df_list[0]['Duration FMJob assigned to AGV [s]']*0+5,
##                hovertext=df_list[0]['hover'],
##                offsetgroup=0,
##                base=df_list[0]['Duration FMJob assigned to AGV [s]'],
##                orientation='v',
##                marker_color='rgb(0,0,255)',
##                showlegend=True),
##                
##                go.Bar(
##                name='Run 1',
##                x=df_list[0]['title+'],
##                y=df_list[0]['Lead time [s]']-df_list[0]['Duration FMJob assigned to AGV [s]']-5,
##                hovertext=df_list[0]['hover'],
##                offsetgroup=0,
##                base=df_list[0]['Duration FMJob assigned to AGV [s]']+5,
##                marker_color=colors[0],
##                orientation='v',
##                showlegend=False),
##                
##                go.Bar(
##                name='Lead time',
##                x=df_list[0]['title+'],
##                y=df_list[0]['Duration FMJob assigned to AGV [s]']*0+5,
##                hovertext=df_list[0]['hover'],
##                offsetgroup=0,
##                base=df_list[0]['Lead time [s]'],
##                orientation='v',
##                marker_color='rgb(255,0,0)',
##                showlegend=True),
##                
##                go.Bar(
##                name=label_list[0],
##                x=df_list[0]['title+'],
##                y=df_list[0]['Lead time deviation [s]']*(-1)-5,
##                hovertext=df_list[0]['hover'],
##                offsetgroup=0,
##                base=df_list[0]['Lead time [s]']+5,
##                orientation='v',
##                marker_color=colors[0]
##                )]
##
##            for dat in range(len(label_list)-1):
##                data.extend([go.Bar(
##                name='Run 1',
##                x=df_list[dat+1]['title+'],
##                y=df_list[dat+1]['Duration FMJob assigned to AGV [s]'],
##                hovertext=df_list[dat+1]['hover'],
##                offsetgroup=dat+1,
##                #base=jis_df["sim time (min)"],
##                orientation='v',
##                marker_color=colors[dat+1],
##                showlegend=False),
##                
##                go.Bar(
##                name='Assigned to AGV',
##                x=df_list[dat+1]['title+'],
##                y=df_list[dat+1]['Duration FMJob assigned to AGV [s]']*0+5,
##                hovertext=df_list[dat+1]['hover'],
##                offsetgroup=dat+1,
##                base=df_list[dat+1]['Duration FMJob assigned to AGV [s]'],
##                orientation='v',
##                marker_color='rgb(0,0,255)',
##                showlegend=False),
##                
##                go.Bar(
##                name='Run 1',
##                x=df_list[dat+1]['title+'],
##                y=df_list[dat+1]['Lead time [s]']-df_list[dat+1]['Duration FMJob assigned to AGV [s]']-5,
##                hovertext=df_list[dat+1]['hover'],
##                offsetgroup=dat+1,
##                base=df_list[dat+1]['Duration FMJob assigned to AGV [s]']+5,
##                marker_color=colors[dat+1],
##                orientation='v',
##                showlegend=False),
##                
##                go.Bar(
##                name='Lead time',
##                x=df_list[dat+1]['title+'],
##                y=df_list[dat+1]['Duration FMJob assigned to AGV [s]']*0+5,
##                hovertext=df_list[dat+1]['hover'],
##                offsetgroup=dat+1,
##                base=df_list[dat+1]['Lead time [s]'],
##                orientation='v',
##                marker_color='rgb(255,0,0)',
##                showlegend=False),
##                
##                go.Bar(
##                name=label_list[dat+1],
##                x=df_list[dat+1]['title+'],
##                y=df_list[dat+1]['Lead time deviation [s]']*(-1)-5,
##                hovertext=df_list[dat+1]['hover'],
##                offsetgroup=dat+1,
##                base=df_list[dat+1]['Lead time [s]']+5,
##                orientation='v',
##                marker_color=colors[dat+1]
##                )])
##
##
##            fig = go.Figure(data=data)
##            fig.update_layout(
##                          
##            #xaxis_title='job index',
##            yaxis_title='time [s]',
##            font_family='IBM Plex Sans',
##            title='Lead Time Deviation Comparison - Delayed Jobs')
##            fig.update_layout({
##            'plot_bgcolor': 'rgb(255, 255,255)',
##            #'paper_bgcolor': 'rgb(1, 10, 51)',
##            })
##            fig.update_layout(font_color="black")
##            return fig
##
##
##    if comp=='processing times':
##        colors=['#68BABB','#605A79','#000000','#E7E8E8','#6DACC0']
##        bd_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##                    
##        for dat in range(len(label_list)):
##            bd_list=übergabe(df_list[dat],label_list[dat],bd_list)
##
##        df_fig = pd.DataFrame(
##        bd_list, columns=['Task','Start','Finish', 'Ressource']
##        )
##        #df_fig_start=df_fig['Start'].min()
##        #df_fig_start=df_fig_start.timestamp()
##        #print(df_fig_start)
##        #print(xxx)
##        #df_fig['Start']=df_fig['Start']-df_fig_start#+1640991600
##        #df_fig['Finish']=df_fig['Start']-df_fig_start#+1640991600
##
##        fig = px.timeline(df_fig, x_start="Start", x_end="Finish", y="Task", color='Ressource')
##        fig.update_layout(
##                          
##            #xaxis_title='job index',
##            yaxis_title='Übergabe location - utilization[%]',
##            font_family='IBM Plex Sans',
##            title='Übergabe Location Utilization')
##        fig.update_layout({
##            'plot_bgcolor': 'rgb(255, 255,255)',
##            #'paper_bgcolor': 'rgb(1, 10, 51)',
##            })
##        fig.update_layout(font_color="black")
##
##        return fig
##
##
##    if comp=='jobs per hour':
##        colors=['#68BABB','#605A79','#000000','#E7E8E8','#6DACC0']
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_aufteilung[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig = go.Figure()
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["langsamdreher"],
##            name='Langsamdreher',
##            marker_color='green',
##            offsetgroup=0,
##
##        ))
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["schnelldreher"],
##            base=df_list[0]["langsamdreher"],
##            name='Schnelldreher',
##            marker_color='red',
##            offsetgroup=0,
##
##        ))
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["specials"],
##            base=df_list[0]["langsamdreher"]+df_list[0]["schnelldreher"],
##            name='Specials',
##            marker_color='blue',
##            offsetgroup=0,
##
##        ))
##
##        for b in range(len(df_list)):
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["langsamdreher"],
##                name=label_list[b],
##                marker_color='green',
##                showlegend=False,
##                offsetgroup=b,
##                marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["schnelldreher"],
##                base=df_list[b]["langsamdreher"],
##                name=label_list[b],
##                marker_color='red',
##                showlegend=False,
##                offsetgroup=b,marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["specials"],
##                base=df_list[b]["langsamdreher"]+df_list[b]["schnelldreher"],
##                name=label_list[b],
##                marker_color='blue',
##                offsetgroup=b,marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##
##        fig.update_layout(font_family='IBM Plex Sans',                          
##        #xaxis_title='job index',
##        xaxis_title='simtime',
##        yaxis_title='FMjobs [#]',
##        title='Breakdown FMJobs in Hour Intervals',)
##        fig.update_layout({
##        'plot_bgcolor': 'rgb(255, 255,255)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##
##        return fig
##
##
##    if comp=='jobs per half hour':
##        colors=['#68BABB','#605A79','#000000','#E7E8E8','#6DACC0']
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append(Dataframes_aufteilung_half[b])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##
##        fig = go.Figure()
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["langsamdreher"],
##            name='Langsamdreher',
##            marker_color='green',
##            offsetgroup=0,
##
##        ))
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["schnelldreher"],
##            base=df_list[0]["langsamdreher"],
##            name='Schnelldreher',
##            marker_color='red',
##            offsetgroup=0,
##
##        ))
##        fig.add_trace(go.Bar(
##            x=df_list[0]['section'], 
##            y=df_list[0]["specials"],
##            base=df_list[0]["langsamdreher"]+df_list[0]["schnelldreher"],
##            name='Specials',
##            marker_color='blue',
##            offsetgroup=0,
##
##        ))
##
##        for b in range(len(df_list)):
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["langsamdreher"],
##                name=label_list[b],
##                marker_color='green',
##                showlegend=False,
##                offsetgroup=b,
##                marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["schnelldreher"],
##                base=df_list[b]["langsamdreher"],
##                name=label_list[b],
##                marker_color='red',
##                showlegend=False,
##                offsetgroup=b,marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##            fig.add_trace(go.Bar(
##                x=df_list[b]['section'], 
##                y=df_list[b]["specials"],
##                base=df_list[b]["langsamdreher"]+df_list[b]["schnelldreher"],
##                name=label_list[b],
##                marker_color='blue',
##                offsetgroup=b,marker_line_color=colors[b],
##                marker_line_width=3,
##
##            ))
##
##        fig.update_layout(  font_family='IBM Plex Sans',                        
##        #xaxis_title='job index',
##        xaxis_title='simtime',
##        yaxis_title='FMjobs [#]',
##        title='Breakdown FMJobs in Half Hour Intervals',)
##        fig.update_layout({
##        'plot_bgcolor': 'rgb(255, 255,255)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##
##        return fig
##
##
##    if comp=='AGV utilization':
##        colors=['#68BABB','#605A79','#000000','#E7E8E8','#6DACC0']
##        bd_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(run[i])
##                    df_list.append([Dataframes_agv[b],Dataframes[b]])
##                    #hover_list.append(Dataframes_edited[b])
##                    label_list.append(Runs[b])
##                    
##        for dat in range(len(label_list)):
##            agv_list=agv_timeline(df_list[dat][0],label_list[dat],bd_list,df_list[dat][1])
##
##        df_fig = pd.DataFrame(
##        agv_list, columns=['AGV','Run','Hover', 'Start', 'Finish']
##        )
##        #df_fig_start=df_fig['Start'].min()
##        #df_fig_start=df_fig_start.timestamp()
##        #print(df_fig_start)
##        #print(xxx)
##        #df_fig['Start']=df_fig['Start']-df_fig_start#+1640991600
##        #df_fig['Finish']=df_fig['Start']-df_fig_start#+1640991600
##        print(df_fig)
##        fig = px.timeline(df_fig, x_start="Start", x_end="Finish", y="AGV", color='Run', hover_name='Hover')
##        fig.update_layout(
##                          
##            #xaxis_title='job index',
##            yaxis_title='AGVs - utilization[%]',
##            font_family='IBM Plex Sans',
##            title='AGV Utilization')
##        fig.update_layout({
##            'plot_bgcolor': 'rgb(255, 255,255)',
##            #'paper_bgcolor': 'rgb(1, 10, 51)',
##            })
##        fig.update_layout(font_color="black")
##
##        return fig
##
##    if comp=='first target':
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    #print(run[i])
##                    df_list.append(Dataframes_first[b])
##                    label_list.append(Runs[b])
##        fig = go.Figure()
##
##        for i in range(len(label_list)):
##            print(df_list[i])
##            fig.add_trace(go.Scatter(x=df_list[i]['x-values'], y=df_list[i]['means'],mode='lines',name=label_list[i]))
##        
##        fig.update_layout(
##            xaxis_title='simtime',
##            yaxis_title='time [s]',
##            title='Average Reach First Target',
##        font_family='IBM Plex Sans')
##        #print('2')
##        fig.update_layout({
##        #'plot_bgcolor': 'rgb(1, 10,51)',
##        #'paper_bgcolor': 'rgb(1, 10, 51)',
##        })
##        fig.update_layout(font_color="black")
##        #print('3')
##        
##        return fig
##
##        
##
##






        
##@app.callback(
##    Output('run', 'multi'),
##    Input('comparison','value'))
##def set_multi(pre_comparison):
##    if pre_comparison=='Jis-wagon development' or pre_comparison=='station utilization' or pre_comparison=='worker utilization':
##        return False
##    else:
##        return True
##@app.callback(
##    Output('comparison','options'),
##    Input('pre-comparison','value'))
##def set_comparison(pre_comparison):
##    print(all_comparison[pre_comparison])
##    return [{'label': i, 'value': i} for i in all_comparison[pre_comparison]]
##
##
##
##
##@app.callback(
##    Output('graph','figure'),
##    Input('run','value'),
##    Input('comparison','value'),
##    Input('pre-comparison','value'))
##def update_graph(run,comp,pre_comp):
##    if pre_comp=='product swirl':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes[i]
##                    line_name=Runs[i]
##            print(line_name)
##            fig=go.Figure()
##            fig.add_trace(go.Scatter(x=df['finish index'],y=df[comp],mode='lines',name=line_name))
##            fig.update_layout(
##                xaxis_title='finish index',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            return fig
##
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Scatter(x=df_list[i]['finish index'], y=df_list[i][comp],mode='lines',name=label_list[i]))
##            fig.update_layout(
##                xaxis_title='finish index',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            return fig
##        
##    if comp=='Jis-wagon duration (min) - hist':
##        df_list=list()
##        value_list=list()
##        if type(run)==str:
##            fig=px.histogram(x=Dataframes_Jis[0]['Jis-wagon duration (min)'],nbins=45)
##
##            
##        
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.extend(Dataframes_Jis[b]['Jis-wagon duration (min)'].values.tolist())
##                        value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['Jis-wagon duration (min)']))
##
##            df=pd.DataFrame(dict(
##                series=value_list,
##                data=df_list))
##        
##        
##            fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=45)
##            
##        fig.update_layout(
##                xaxis_title='Jis-wagon duration (min)',
##                yaxis_title='count',
##        font_family='IBM Plex Sans')
##            
##            
##    if comp=='Jis-wagon development':
##        for i in range(len(Runs)):
##            if run==Runs[i]:
##                jis_df=Dataframes_Jis[i]
##        fig = go.Figure(
##            data=[
##                go.Bar(
##                name='Jis-wagon duration (min)',
##                y=jis_df['Jis-wagon nr'],
##                x=jis_df['Jis-wagon duration (min)'],
##                offsetgroup=0,
##                base=jis_df["sim time (min)"],
##                orientation='h'
##                ),
##                go.Bar(name='Time Intervall',
##                y=jis_df['Jis-wagon nr'],
##                x=jis_df['time intervall (min)'],
##                base=(jis_df['Jis-wagon duration (min)']+jis_df['sim time (min)']),
##                offsetgroup=1,
##                orientation='h'
##                )
##            ],
##            layout=go.Layout(
##                yaxis_title='Time in min',
##                xaxis_title='JIS-wagon',
##            font_family='IBM Plex Sans'))
##        fig['layout']['yaxis']['autorange'] = "reversed"
##        
##        
##        
##    
##    if comp=='time intervall (min) - hist':
##        df_list=list()
##        value_list=list()
##        if type(run)==str:
##            fig=px.histogram(x=Dataframes_Jis[0]['time intervall (min)'],nbins=65)
##
##            
##        
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.extend(Dataframes_Jis[b]['time intervall (min)'].values.tolist())
##                        value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['time intervall (min)']))
##
##            df=pd.DataFrame(dict(
##                series=value_list,
##                data=df_list))
##        
##        
##            fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=65)
##            
##        fig.update_layout(
##                xaxis_title='time intervall (min) - hist',
##                yaxis_title='count',
##        font_family='IBM Plex Sans')
##            
##    if comp=='time intervall (min)':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes_Jis[i]
##                    line_name=Runs[i]
##            print(line_name)
##            fig=go.Figure()
##            fig.add_trace(go.Bar(x=df['Jis-wagon nr'],y=df['time intervall (min)'],name=line_name))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                xaxis_title='Jis-wagon nr',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##            return fig
##
##        else:
##            print('list')
##            print(len(run))
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes_Jis[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            print(len(label_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['time intervall (min)'],name=label_list[i]))
##            fig.update_layout(barmode='group')
##            fig.update_layout(
##                xaxis_title='Jis-wagon nr',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##
##            return fig
##        
##    if comp=='Jis-wagon duration (min)':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes_Jis[i]
##                    line_name=Runs[i]
##            print(line_name)
##            fig=go.Figure()
##            fig.add_trace(go.Bar(x=df['Jis-wagon nr'],y=df['Jis-wagon duration (min)'],name=line_name))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                xaxis_title='Jis-wagon nr',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##            return fig
##
##        else:
##            print('list')
##            print(len(run))
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes_Jis[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['Jis-wagon duration (min)'],name=label_list[i]))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                xaxis_title='Jis-wagon nr',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##    
##    if comp=='simultaneous Jis-wagon':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=simul(Dataframes_Jis[i])
##                    line_name=Runs[i]
##                    
##
##                    
##            fig=go.Figure()
##            fig.add_trace(go.Scatter(y=df['simul'],mode='lines',name=line_name))
##            
##            fig.update_layout(
##                xaxis_title='time (sec)',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            return fig
##
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(simul(Dataframes_Jis[b]))
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Scatter(y=df_list[i]['simul'],mode='lines',name=label_list[i]))
##                
##            fig.update_layout(
##                xaxis_title='time (sec)',
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##
##            return fig
##        
##        
##    if comp=='station percentage active':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes_Station[i]
##                    line_name=Runs[i]
##            fig=go.Figure()
##            fig.add_trace(go.Bar(x=df['station name'],y=df['percentage active'],name=line_name))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##            return fig
##
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes_Station[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage active'],name=label_list[i]))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##
##            return fig
##        
##        
##    if comp=='station percentage active working':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes_Station[i]
##                    line_name=Runs[i]
##            fig=go.Figure()
##            fig.add_trace(go.Bar(x=df['station name'],y=df['percentage working active'],name=line_name))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##            return fig
##
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes_Station[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage working active'],name=label_list[i]))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##
##            return fig
##        
##    if comp=='worker percentage working':
##        df_list=list()
##        label_list=list()
##        if type(run)==str:
##            for i in range(len(Runs)):
##                if run==Runs[i]:
##                    df=Dataframes_Worker[i]
##                    line_name=Runs[i]
##            fig=go.Figure()
##            fig.add_trace(go.Bar(x=df['worker name'],y=df['percentage working'],name=line_name))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##            font_family='IBM Plex Sans')
##            
##            return fig
##
##        else:
##            for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.append(Dataframes_Worker[b])
##                        label_list.append(Runs[b])
##
##
##            print(len(df_list))
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Bar(x=df_list[i]['worker name'], y=df_list[i]['percentage working'],name=label_list[i]))
##            fig.update_layout(barmode='group')
##            
##            fig.update_layout(
##                yaxis_title=comp,
##                font_family='IBM Plex Sans')
##
##            return fig
##        
##    
##    if comp=='station utilization':
##        for i in range(len(Runs)):
##            if run==Runs[i]:
##                df=pd.DataFrame(utilization_list[i])
##                
##        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
##                      group_tasks=True)
##        
##        
##        
##        
##    if comp=='worker utilization':
##        for i in range(len(Runs)):
##            if run==Runs[i]:
##                df=pd.DataFrame(utilization_list_worker[i])
##                
##        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
##                      group_tasks=True)
##        
##        
##    return fig  
##
##
##@app.callback(
##    [Output('data-table','columns'),
##    Output('data-table','data')],
##    [Input('run','value')])
##def update_data_table(run):
##    if type(run)==str:
##        run_list=list()
##        run_list.append(run)
##        run=run_list
##    else:
##        rund_list=run
##
##        
##    name_list=list()
##    num_jobs=list()
##    num_agvs=list()
##    average_list_ut=list()
##    agv_assigned=list()
##    first_target=list()
##    fm_duration=list()
##    waiting_time=list()
##    on_time=list()
##   # average_list_anfahrt=list()
##    #average_list_process=list()
##    #average_list_target1=list()
##    #average_list_schnelldreher=list()
##    #average_list_leadtime=list()
##    for i in range(len(run)):
##        for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(avg_list)
##                    name_list.append(Runs[b])
##                    num_jobs.append(avg_list[b][1])
##                    num_agvs.append(avg_list[b][7])
##                    average_list_ut.append(avg_list[b][4])
##                    agv_assigned.append(avg_list[b][0])
##                    first_target.append(avg_list[b][8])
##                    fm_duration.append(avg_list[b][3])
##                    #waiting_time.append(avg_list[b][2])
##                    on_time.append(avg_list[b][11])
##                    #average_list_ut.append(round(Dataframes_ut[b].mean(),dec))
##                    #average_list_anfahrt.append(round(Dataframes_anfahrt[b].mean(),dec))
##                    #average_list_process.append(round(Dataframes_process[b].mean(),dec))
##                    #average_list_target1.append(round(Dataframes_target1[b].mean(),dec))
##                    #average_list_schnelldreher.append(round(Dataframes_schnelldreher[b].mean(),dec))
##                    #average_list_leadtime.append(round(Dataframes_leadtime[b].mean(),dec))
##    
##    
##        data={'RUN':name_list,
##              'FM Jobs[#]':num_jobs,
##              'AGVs [#]':num_agvs,
##              'Avg AGV utilization [%]':average_list_ut,
##              'Avg time FMJob assigned to AGV [s]':agv_assigned,
##              'Avg duration reach first target [s]':first_target,
##              'Avg FMJob duration [s]':fm_duration,
##              #'Avg waiting time [%]':waiting_time,
##              'FMJobs on time [%]':on_time,
##              
        
          #'average anfahrt':average_list_anfahrt,
          #'average process time':average_list_process,
          #'average target1':average_list_target1,
          #'average schnelldreher':average_list_schnelldreher,
          #'average leadtime difference':average_list_leadtime,
##              }
##    
##
##
##    
##    if pre_comp=='product swirl':
##        name_list=list()
##        average_list_prate=list()
##        average_list_pratehour=list()
##        average_list_pratehour_init=list()
##        average_list_prateworker=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_prate.append(round(Dataframes[b]['p-rate (1/min)'].mean(),dec))
##                        average_list_pratehour.append(round(Dataframes[b]['p-rate hour (1/min)'].mean(),dec))
##                        average_list_pratehour_init.append(round(Dataframes[b]['p-rate hour (1/min)'][200:-1].mean(),dec))
##                        average_list_prateworker.append(round(Dataframes[b]['p-rate per worker (1/(min/worker))'].mean(),dec))
##    
##    
##        data={'run':name_list,
##          'p-rate   ':average_list_prate,
##          'p-rate hour  ':average_list_pratehour,
##          'p-rate hour after 200 jobs':average_list_pratehour_init,
##          'p-rate per worker': average_list_prateworker}
##        
##        
##    if pre_comp=='Jis-wagon':
##        name_list=list()
##        average_list_duration=list()
##        average_list_timeintervall=list()
##        average_list_simul=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_duration.append(round(Dataframes_Jis[b]['Jis-wagon duration (min)'].mean(),dec))
##                        average_list_timeintervall.append(round(Dataframes_Jis[b]['time intervall (min)'].mean(),dec))
##                        average_list_simul.append(round(simul(Dataframes_Jis[b]).mean(),dec))
##    
##    
##        data={'run':name_list,
##          'Jis-wagon duration (min)':average_list_duration,
##          'time intervall (min)':average_list_timeintervall,
##          'simultaneous Jis-wagon':average_list_simul}
##        
##        
##    if pre_comp=='utilization station':
##        name_list=list()
##        average_list_active=list()
##        average_list_working=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_active.append(round(Dataframes_Station[b]['percentage active'].mean(),dec))
##                        average_list_working.append(round(Dataframes_Station[b]['percentage working active'].mean(),dec))
##    
##    
##        data={'run':name_list,
##          'percentage active':average_list_active,
##          'percentage working active':average_list_working}
##        
##        
##    if pre_comp=='run information':
##        name_list=list()
##        amr_list=list()
##        program_list=list()
##        worker_list=list()
##        site=list()
##        cycle_time=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    print(len(info_df))
##                    for d in range(len(info_df)):
##                         if info_df.loc[d,'Run ID']==Runs[b]:
##                                correct_value=d
##                                
##                    amr_list.append(info_df.loc[correct_value,'AMRs'])
##                    name_list.append(Runs[b])
##                    program_list.append(info_df.loc[correct_value,'Programm'])
##                    worker_list.append(info_df.loc[correct_value,'Number of Workers'])
##                    site.append(info_df.loc[correct_value,'site'])
##                    cycle_time.append(info_df.loc[correct_value,'Rotation cycle time'])
##                
##        data={'run':name_list,
##          'number of AMRs':amr_list,
##             'program':program_list,
##             'number of workers':worker_list,
##             'site':site,
##            'rotation cycle time':cycle_time}
##                    
##    
##
##    df = pd.DataFrame(data)
##        
##    columns=[{"name": i, "id": i} 
##                 for i in df.columns]
##    
##    data=df.to_dict('records')
##    print('55')
##    
##    return columns,data
##
##@app.callback(
##    Output('placeholder2','children'),
##    Input('run','value'))
##def del_success(value):
##        if value!=run_global_list:
##            return ' '
##        
##        else:
##            return 'Suc'
##
##
##
##@app.callback(
##    Output('placeholder','children'),
##    [Input('run','value'),
##    Input('pre-comparison','value'),
##    Input('download','value')])
##
##
##def create_pdf(run,pre_comp,dropdown):
##    if dropdown=='Download':
##        if type(run)==str:
##            run_list=list()
##            run_list.append(run)
##            run=run_list
##            
##
##        import os
##        #if not os.path.exists('fig_folder'):
##        #    os.makedirs('fig_folder')
##
##
##        #Initialize figure list   
##        figure_list=list()
##
##        figures=['p-rate per worker (1/(min/worker))', 'p-rate hour (1/min)', 'p-rate hour average 10 (1/min)','p-rate (1/min)']
##        texts=['p-rateworker', 'p-ratehour', 'p-rateaverage10','p-rate']
##        rounds=0
##        for figure in figures:
##            df_list=list()
##            label_list=list()
##            for i in range(len(run)):
##                    for b in range(len(Runs)):
##                        if run[i]==Runs[b]:
##                            df_list.append(Dataframes[b])
##                            label_list.append(Runs[b])
##
##
##            fig = go.Figure()
##            for i in range(len(label_list)):
##                fig.add_trace(go.Scatter(x=df_list[i]['finish index'], y=df_list[i][figure],mode='lines',name=label_list[i]))
##            fig.update_layout(
##                xaxis_title='finish index',
##                yaxis_title=figure,
##                font_family='IBM Plex Sans')
##
##            fig.write_html("04_Downloads/04_01_Figures/"+texts[rounds]+".html")
##            pio.write_image(fig, "04_Downloads/04_01_Figures/"+texts[rounds]+".svg")
##            rounds+=1
##            
##        df_list=list()
##        value_list=list()   
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        df_list.extend(Dataframes_Jis[b]['Jis-wagon duration (min)'].values.tolist())
##                        value_list.extend([Runs[b]]*len(Dataframes_Jis[b]['Jis-wagon duration (min)']))
##
##                df=pd.DataFrame(dict(
##                series=value_list,
##                data=df_list))
##        
##        
##        fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=45)
##            
##        fig.update_layout(
##                xaxis_title='Jis-wagon duration (min)',
##                yaxis_title='count',
##        font_family='IBM Plex Sans')
##        
##        fig.write_html("04_Downloads/04_01_Figures/Jis-duration-hist.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/Jis-duration-hist.svg")
##        
##        
##        
##        df_list=list()
##        value_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.extend(Dataframes_Jis[b]['time intervall (min)'].values.tolist())
##                    value_list.extend([Runs[b]]*len(Dataframes_Jis[b]['time intervall (min)']))
##
##        df=pd.DataFrame(dict(
##                series=value_list,
##                data=df_list))
##        
##        
##        fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=65)
##            
##        fig.update_layout(
##                xaxis_title='time intervall (min) - hist',
##                yaxis_title='count',
##        font_family='IBM Plex Sans')
##        
##        fig.write_html("04_Downloads/04_01_Figures/time-intervall-hist.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/time-intervall-hist.svg")
##        
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(Dataframes_Jis[b])
##                    label_list.append(Runs[b])
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['time intervall (min)'],name=label_list[i]))
##        fig.update_layout(barmode='group')
##        fig.update_layout(
##            xaxis_title='Jis-wagon nr',
##            yaxis_title='time-intervall (min)',
##            font_family='IBM Plex Sans')
##        
##        
##        pio.write_image(fig, "04_Downloads/04_01_Figures/time-intervall.svg")
##        fig.write_html("04_Downloads/04_01_Figures/time-intervall.html")
##        
##        
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(Dataframes_Jis[b])
##                    label_list.append(Runs[b])
##
##
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['Jis-wagon duration (min)'],name=label_list[i]))
##        fig.update_layout(barmode='group')
##            
##        fig.update_layout(
##                xaxis_title='Jis-wagon nr',
##                yaxis_title='Jis-wagon duration (min)',
##            font_family='IBM Plex Sans')
##        
##        
##        fig.write_html("04_Downloads/04_01_Figures/Jis-duration.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/Jis-duration.svg")
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(simul(Dataframes_Jis[b]))
##                    label_list.append(Runs[b])
##
##
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Scatter(y=df_list[i]['simul'],mode='lines',name=label_list[i]))
##                
##        fig.update_layout(
##                xaxis_title='time (sec)',
##                yaxis_title='simultenueous Jis-wagon',
##            font_family='IBM Plex Sans')
##
##        fig.write_html("04_Downloads/04_01_Figures/simul.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/simul.svg")
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(Dataframes_Station[b])
##                    label_list.append(Runs[b])
##
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage active'],name=label_list[i]))
##        fig.update_layout(barmode='group')
##            
##        fig.update_layout(
##                yaxis_title='station percentage active',
##            font_family='IBM Plex Sans')
##        
##        fig.write_html("04_Downloads/04_01_Figures/station-active.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/station-active.svg")
##        
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(Dataframes_Station[b])
##                    label_list.append(Runs[b])
##
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage working active'],name=label_list[i]))
##        fig.update_layout(barmode='group')
##            
##        fig.update_layout(
##                yaxis_title='station percentage active working',
##            font_family='IBM Plex Sans')
##        
##        fig.write_html("04_Downloads/04_01_Figures/station-active-working.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/station-active-working.svg")
##        
##        
##        df_list=list()
##        label_list=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    df_list.append(Dataframes_Worker[b])
##                    label_list.append(Runs[b])
##
##
##        fig = go.Figure()
##        for i in range(len(label_list)):
##            fig.add_trace(go.Bar(x=df_list[i]['worker name'], y=df_list[i]['percentage working'],name=label_list[i]))
##        fig.update_layout(barmode='group')
##            
##        fig.update_layout(
##                yaxis_title='worker percentage working',
##                font_family='IBM Plex Sans')
##        
##        
##        fig.write_html("04_Downloads/04_01_Figures/worker-working.html")
##        pio.write_image(fig, "04_Downloads/04_01_Figures/working-working.svg")
##        
##        #Save the tables##########################################
##        #if not os.path.exists('table_folder'):
##         #   os.makedirs('table_folder')
##            
##        name_list=list()
##        average_list_working=list()
##        average_list_idle=list()
##        average_list_rotation=list()
##        average_list_rotations=list()
##        print('utilization worker')
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_working.append(round(Dataframes_Worker[b]['percentage working'].mean(),dec))
##                        average_list_idle.append(round(Dataframes_Worker[b]['percentage idle'].mean(),dec))
##                        average_list_rotation.append(round(Dataframes_Worker[b]['percentage rotation'].mean(),dec))
##                        average_list_rotations.append(round(Dataframes_Worker[b]['average rotations (1/h)'].mean(),dec))
##    
##    
##        data={'run':name_list,
##          'percentage working':average_list_working,
##          'percentage idle':average_list_idle,
##          'percentage rotation':average_list_rotation,
##          'average rotations (1/h)':average_list_rotations}
##        
##        writer = pd.ExcelWriter('04_Downloads/04_02_Tables/utilization-worker.xlsx')
##        
##        pd.DataFrame(data).to_excel(writer,index=False)
##        
##        writer.save()
##        
##        name_list=list()
##        average_list_prate=list()
##        average_list_pratehour=list()
##        average_list_pratehour_init=list()
##        average_list_prateworker=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_prate.append(round(Dataframes[b]['p-rate (1/min)'].mean(),dec))
##                        average_list_pratehour.append(round(Dataframes[b]['p-rate hour (1/min)'].mean(),dec))
##                        average_list_pratehour_init.append(round(Dataframes[b]['p-rate hour (1/min)'][200:-1].mean(),dec))
##                        average_list_prateworker.append(round(Dataframes[b]['p-rate per worker (1/(min/worker))'].mean(),dec))
##    
##    
##        data={'run':name_list,
##          'p-rate':average_list_prate,
##          'p-rate hour':average_list_pratehour,
##          'p-rate hour after 200 jobs':average_list_pratehour_init,
##          'p-rate per worker': average_list_prateworker}
##        
##        
##        writer = pd.ExcelWriter('04_Downloads/04_02_Tables/product-swirl.xlsx')
##        
##        pd.DataFrame(data).to_excel(writer,index=False)
##        
##        writer.save()
##        
##        
##        name_list=list()
##        average_list_duration=list()
##        average_list_timeintervall=list()
##        average_list_simul=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_duration.append(round(Dataframes_Jis[b]['Jis-wagon duration (min)'].mean(),dec))
##                        average_list_timeintervall.append(round(Dataframes_Jis[b]['time intervall (min)'].mean(),dec))
##                        average_list_simul.append(round(simul(Dataframes_Jis[b]).mean(),dec))
##    
##    
##        data={'run':name_list,
##          'Jis-wagon duration (min)':average_list_duration,
##          'time intervall (min)':average_list_timeintervall,
##          'simultaneous Jis-wagon':average_list_simul}
##        
##        
##        writer = pd.ExcelWriter('04_Downloads/04_02_Tables/Jis-wagon.xlsx')
##        
##        pd.DataFrame(data).to_excel(writer, index=False)
##        
##        writer.save()
##        
##        name_list=list()
##        average_list_active=list()
##        average_list_working=list()
##        for i in range(len(run)):
##                for b in range(len(Runs)):
##                    if run[i]==Runs[b]:
##                        name_list.append(Runs[b])
##                        average_list_active.append(round(Dataframes_Station[b]['percentage active'].mean(),dec))
##                        average_list_working.append(round(Dataframes_Station[b]['percentage working active'].mean(),dec))
##    
##    
##        data={'run':name_list,
##          'percentage active':average_list_active,
##          'percentage working active':average_list_working}
##        
##        
##        writer = pd.ExcelWriter('04_Downloads/04_02_Tables/utilization-station.xlsx')
##        
##        pd.DataFrame(data).to_excel(writer,index=False)
##        
##        writer.save()
##        
##        
##        name_list=list()
##        amr_list=list()
##        program_list=list()
##        worker_list=list()
##        site=list()
##        cycle_time=list()
##        for i in range(len(run)):
##            for b in range(len(Runs)):
##                if run[i]==Runs[b]:
##                    for d in range(len(info_df)):
##                         if info_df.loc[d,'Run ID']==Runs[b]:
##                                correct_value=d
##                                
##                    amr_list.append(info_df.loc[correct_value,'AMRs'])
##                    name_list.append(Runs[b])
##                    program_list.append(info_df.loc[correct_value,'Programm'])
##                    worker_list.append(info_df.loc[correct_value,'Number of Workers'])
##                    site.append(info_df.loc[correct_value,'Worker Rotation'])
##                    cycle_time.append(info_df.loc[correct_value,'Rotation cycle time'])
##                    
##        print(cycle_time)
##                
##        data={'run':name_list,
##          'number of AMRs':amr_list,
##             'program':program_list,
##             'number of workers':worker_list,
##             'site':site,
##             'rotation cycle time':cycle_time}
##        
##        
##        writer = pd.ExcelWriter('04_Downloads/04_02_Tables/run-information.xlsx')
##        
##        pd.DataFrame(data).to_excel(writer, index=False)
##        
##        writer.save()
##
##        return ('Downloaded the comparison of: ' +str(run))
##    
##    else: 
##        return ' '
##    
##        
port_rand = random.randint(100, 900)    
    
def open_browser():
      webbrowser.open_new('http://127.0.0.1:'+str(port_rand)+'/')
    
        
        
        
        
        
        
        


   

if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run_server(host='127.0.0.1',port=port_rand,debug=False)
    #app.run_server(debug=False)
