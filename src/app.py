# -*- coding: utf-8 -*-

# Main Libaries
import numpy as np
import pandas as pd

# Datetime Library
from datetime import datetime, timedelta
from datetime import date

# Yahoo Finance API
import yfinance as yf

# Data Visualisation
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.tools as tls

from wordcloud import WordCloud    

# Data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Web App
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash import Dash, dash_table
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Group
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container



# Stats Library
from scipy import stats

from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
)
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint


BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"





app = dash.Dash(external_stylesheets=[BS])
server = app.server

app.config.suppress_callback_exceptions = True


######################################################################################################################

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "1rem 1rem",
    "background-color": "#FDEEF4",
    "margin-left": "0.5rem",
    #"margin-right": "4rem",
    "border": "2px solid black"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    #"margin-left": "4.5rem",
    #"margin-right": "0.5rem",
    "background-color": "#FFFAFA"

}


##################################################### Load & Transform ###############################################

# Pivot DF
df = pd.read_csv("btc_components.csv")
df = df.interpolate()
df = df.dropna()
df = df.drop(columns=["Volume.1", "Returns.1", "Returns"])
# Percent Change + Drop null values
df['returns'] = 100 * df.Close.pct_change().dropna()

# Apply log returns
df['log_returns'] = np.log(df.Close/df.Close.shift(1))

# Drop NA
df = df.dropna()

# Melted DF
df_melted = df.melt(id_vars=['Date'])
df_melted["Date"] = pd.to_datetime(df_melted["Date"])
df_melted["month"] = df_melted["Date"].dt.month
df_melted["year"] = df_melted["Date"].dt.year


# Tweetsdis
tweets_df = pd.read_csv("tweets_sentiment.csv")
tweets_df["Datetime"] = pd.to_datetime(tweets_df["Datetime"])
tweets_df["month"] = tweets_df["Datetime"].dt.month
tweets_df["year"] = tweets_df["Datetime"].dt.year


###################################################### Content ######################################################


sidebar = dbc.Card([
    dbc.CardBody([ 
    html.H2("Menu", style={"fontFamily": "courier",
                           "textAlign": "center"}),
    html.Hr(),
    html.P("", className="lead"),
    dbc.Nav(
        [
                            dbc.Card(
                    dbc.CardBody([    
            dbc.NavLink("Home Page", href="/", active="exact",
                        style={'fontFamily': 'courier',
                               'textAlign':'center',
                               'color': '#000000',
                                "border": "1px solid black",                               
                               "background-color": "#FFFAFA"})
                    
                    ]),style={"border": "2px solid black", "background-color": "#FDEEF4"}), 

                html.Br(),               
                            dbc.Card(
                    dbc.CardBody([    
                dbc.NavLink("Statistical Analysis", href="/page-1", active="exact",
                        style={'fontFamily': 'courier',
                               'textAlign':'center',
                               'color': '#000000',
                                "border": "1px solid black",                               
                               "background-color": "#FFFAFA"})
                    
                    ]),style={"border": "2px solid black", "background-color": "#FDEEF4"}), 
                                
                 html.Br(),                    
   
                            dbc.Card(
                    dbc.CardBody([    
                dbc.NavLink("Components Analysis", href="/page-2", active="exact",
                        style={'fontFamily': 'courier',
                               'textAlign':'center',
                               'color': '#000000',
                                "border": "1px solid black",                               
                               "background-color": "#FFFAFA"})
                    
                    ]),style={"border": "2px solid black", "background-color": "#FDEEF4"}), 
            
                html.Br(),                 
            
                            dbc.Card(
                    dbc.CardBody([    
                dbc.NavLink("Sentiment Analysis", href="/page-3", active="exact",
                        style={'fontFamily': 'courier',
                               'textAlign':'center',
                               'color': '#000000',
                                "border": "1px solid black",                               
                               "background-color": "#FFFAFA"})
                    
                    ]),style={"border": "2px solid black", "background-color": "#FDEEF4"}),  
            
                html.Br(),                 
        
            
            
            
        ], 
        vertical=True, 
        pills=True,
    )
    ])
    
], color="#f0f8ff", style=SIDEBAR_STYLE)


content = html.Div(id='page-content', children=[], style=CONTENT_STYLE)


####################################################### Layout #########################################################

# Initiliase app
app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([
        dbc.Col(sidebar,width=1),
        dbc.Col(content, width=11)
        
        
    ])   
    
])


########################################################### Callbacks #####################################################

########################################################### Home Page #####################################################

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname =="/":
        return dbc.Container([
            
            dbc.Card(
            dbc.CardBody([               
    dbc.Card(
        dbc.CardBody([

            # Header
            dbc.Row([
                dbc.Col([
                html.H2("Descriptive Analysis", style={"fontFamily": "courier",
                                                          "textAlign": "center"})
                ], width=12)]),            
            html.Hr(),
            
            dbc.Row([
                
                
               # 1st Led
                dbc.Col([
                daq.LEDDisplay(
                    id='my-LED-display-1',
                    label="Month",
                    size = 25,
                    color="#000000"
                ),
                    
                # 1st Slider    
                dcc.Slider(
                    id='my-LED-display-slider-1', 
                    min=1,
                    max=12,
                    tooltip={"placement": "bottom", "always_visible": False},
                    step=1,
                    value=8
                )
                ], width=4),
                

                
               # 1st Led
                dbc.Col([
                daq.LEDDisplay(
                    id='my-LED-display-2',
                    label="Year",                 
                    size = 25,
                    color="#000000"
                ),
                    
                # 1st Slider    
                dcc.Slider(
                    id='my-LED-display-slider-2',
                    min=2019,
                    max=2022, 
                    marks={
                    2019: '2019',
                    2020: '2020',
                    2021: '2021',
                    2022: '2022'},
                    tooltip={"placement": "bottom", "always_visible": False},
                    step=1,
                    value=2019
                )
                ], width=4),
                
                
                dbc.Col([
                html.H4("BTC Features", className="card-title", style={"fontFamily": "courier",
                                                                          "textAlign": "center"}),
                        dcc.Dropdown(
                            id='btc-components-dropdown',
                            placeholder="Select  BTC Feature",
                     options=[{'label': i, 'value': i}
                              for i in df_melted['variable'].unique()],
                            style=dict(
                                width='100%',
                                verticalAlign="center",
                                justifyContent = 'center',
                                border= "1px solid black",
                                backgroundColor= "#f0f8ff",                               
                                fontSize=20,                                
                                fontFamliy = 'courier'),
                            value='Close'),
                    
                ], width=4)
            ], align='center'), 
   
            dbc.Row([
                dbc.Col([
                html.H4("Average", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),                     
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='mean_indicator_plot')
                            ],style={'textAlign': 'center'})
                        ]),style={"border": "1px solid black"}),
                ], width=2),
                
               dbc.Col([
                html.H4("Min", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),                    
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='min_indicator_plot')
                            ],style={'textAlign': 'center'})
                        ]),style={"border": "1px solid black"}),
                ], width=2),
               dbc.Col([
                html.H4("Distribution", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),                    
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='distribution_hist_plot')
                            ],style={'textAlign': 'center'})
                        ]),style={"border": "4px solid black"}),
                ], width=4),                
                
               dbc.Col([
                html.H4("Max", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),                    
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='max_indicator_plot')
                            ],style={'textAlign': 'center'})
                        ]),style={"border": "1px solid black"}),
                ], width=2),
                
                
               dbc.Col([
                html.H4("Std. Dev.", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),                    
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='std_indicator_plot')
                            ],style={'textAlign': 'center'})
                        ]),style={"border": "1px solid black"}),
                ], width=2),
            ], align='center'), 
            
            html.Br(),
            
            dbc.Row([
                
                dbc.Col([
                html.H4("Trend Over Time", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),         
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="areachart_trend_plot")
                            ])
                        ]),style={"border": "1px solid black"}),
                ], width=12)               
                
            ], align='center'),                 
            
            html.Br(),
            
            dbc.Row([
                
                dbc.Col([
                html.H4("Monthly Distribution", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),               
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="distribution_bp_plot_m")
                            ])
                        ]),style={"border": "1px solid black"}),
                ], width=6),
               

                
                
                dbc.Col([
                html.H4("Yearly Distribution", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),              
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="distribution_bp_plot_y")
                            ])
                        ]),style={"border": "1px solid black"}),
                ], width=6), 
                
            ], align='center')
            
        ],style={"border": "1px solid black","background-color": "#FDEEF4",})
    )]),style={"border": "2px solid black","background-color": "#FDEEF4"})
])  
    
   # Page 1
    
    elif pathname == "/page-1":
        return dbc.Container([
            dbc.Card(
            dbc.CardBody([            
    dbc.Card(
        dbc.CardBody([

        # Header       
            dbc.Row([
                dbc.Col([
                html.H2("Statistical Analysis of the BTC Features", style={"fontFamily": "courier",
                                                                           "textAlign": "center"})
                ], width=12)]),
            
            html.Hr(),            
                
            # 1st Row
            dbc.Row([
                dbc.Col([
                    
        dbc.Card(
            dbc.CardBody(
                [
                html.H4("Select Period", className="card-title", 
                        style={"fontFamily": "courier", "textAlign":"center"}),
                    dcc.DatePickerRange(
                        id = 'date-picker-range',
                        min_date_allowed=date(2019, 8, 2),
                        max_date_allowed=date(2022, 7, 31),
                        start_date=date(2022, 1, 1),
                        display_format='DD/MM/YY',                        
                        end_date=date(2022, 7, 31),
                   style = {
                        'font-size': '3px',
                        'display': 'inline-block', 
                        'border-radius' : '0.2px', 
                        'border' : '1px solid #ccc',
                        'color': '#333', 
                        'border-spacing' : '0', 
                        'border-collapse' :'separate',
                        'width':'100%',
                       'padding-left': '0px',
                       'padding-right': '0px',
                       'text-align':'center'                       
                        }                         
                    ),
                ],style = {'width': '100%',"textAlign":"center"}
                
            ),style={"border": "1px solid black","background-color": "#FDEEF4", 'font-family': 'courier', 'width': '38.5%'}),                  
                    
                    
                ], width=9), 
                
                
                dbc.Col([
                html.H4("BTC Features", className="card-title", style={"fontFamily": "courier",
                                                                      }),
                        dcc.Dropdown(
                            id='btc-components-dropdown',
                            placeholder="Select  BTC Feature",
                     options=[{'label': i, 'value': i}
                              for i in df_melted['variable'].unique()],
                            style=dict(
                                width='100%',
                                verticalAlign="center",
                                border= "1px solid black",
                                backgroundColor= "#f0f8ff",                                
                                justifyContent = 'center',
                                fontSize=20,
                                fontFamliy = 'Courier'),
                            value='Gold'),
                    
                ], width=3),                
            ], align='center'), 
            
            html.Br(),
            
            dbc.Row([

                
                dbc.Col([
                html.H4("Correlation", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="correlation_plot")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),
                               
                
                dbc.Col([
                html.H4("Cointegration p-value", className="card-title", 
                        style={"fontFamily": "courier", "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="cointegration_plot")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),  
                
                
                dbc.Col([
                html.H4("Causality p-value", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="causality_plot")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),
                
                dbc.Col([

                ], width=3),
            ], align='center'), 
            
            
            # Breakline
            html.Br(), 
            
            # 2nd Row
            dbc.Row([
                
                dbc.Col([
                html.H4("Correlation Matrix", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),         
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="correlation_matrix_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6),
                
                dbc.Col([
                html.H4("Standardised Trend", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="standardised_trend_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6),
                
              
            ], align='center'), 
                
        html.Br(),
            
            dbc.Row([
                
                dbc.Col([], width=3),
                dbc.Col([
                html.H4("Granger Causality Results", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),         
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='causality_table')  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=12),                            
                
            ], align='center'),


        ],style={"border": "1px solid black","background-color": "#FDEEF4",'width':'73rem'})
    )]),style={"border": "2px solid black","background-color": "#FDEEF4",'width':'75rem'})
])  
    
    # Page 2
    elif pathname == "/page-2":
        return dbc.Container([
            dbc.Card(
            dbc.CardBody([             
       dbc.Card(
        dbc.CardBody([
           
        # Header       
            dbc.Row([
                dbc.Col([
                html.H2("Time Series Analysis", style={"fontFamily": "courier",
                                                       "textAlign": "center"})
                ], width=12)]),
            
            html.Hr(),            
            
            # 1st Row
            dbc.Row([
                
                dbc.Col([
                html.H4("Ad Fuller Test p-value", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="ad_fuller_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=4),    
                
                dbc.Col([], width=2),                 
                
                dbc.Col([
                html.H4("BTC Features", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                        dcc.Dropdown(
                            id='btc-components-dropdown',
                            placeholder="Select  BTC Feature",
                     options=[{'label': i, 'value': i}
                              for i in df_melted['variable'].unique()],
                            style=dict(
                                width='100%',
                                verticalAlign="center",
                                justifyContent = 'center',
                                border= "1px solid black",
                                backgroundColor= "#f0f8ff",                                
                                fontSize=20,                                
                                fontFamliy = 'courier'),
                            value='Close'),
                    
                ], width=6),

            ], align='center'), 
            
            
        html.Br(),
            
            dbc.Row([
                dbc.Col([
                html.H4("Volatility / Returns", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),          
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="volatility_trend_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=12),                
                
                
            ], align='center'), 
            
            html.Br(),
            
            # 3rd Row
            dbc.Row([  
                dbc.Col([
                html.H4("PACF / ACF", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),          
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="pacf_acf_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6),
                
                dbc.Col([
                html.H4("Seasonal Components", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),            
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="seasonal_plots")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6),
            ], align='center'), 

     
        ],style={"border": "1px solid black","background-color": "#FDEEF4",'width':'73rem'})
    )]),style={"border": "2px solid black","background-color": "#FDEEF4",'width':'75rem'})
])  
    
    
    
    # Page 3
    elif pathname == "/page-3":
        return dbc.Container([
            
            dbc.Card(
            dbc.CardBody([              
       dbc.Card(
        dbc.CardBody([
           
        # Header       
            dbc.Row([
                dbc.Col([
                html.H2("Sentiment Analysis", style={"fontFamily": "courier",
                                                       "textAlign": "center"})
                ], width=12)]),
            
            html.Hr(),            
            
            # 1st Row
            dbc.Row([
                
                dbc.Col([
                html.H4("Min Year", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),         
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='min_year_plot') 
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=2),  
            
                 
                dbc.Col([
                html.H4("Max Year", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),         
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id='max_year_plot')  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=2),
                
                
               # 1st Led
                dbc.Col([
                daq.LEDDisplay(
                    id='my-LED-display-2',
                    label="Year",
                 
                    size = 25,
                    value='1.001',
                    color="#000000"
                ),
                    
                # 1st Slider    
                dcc.Slider(
                    id='my-LED-display-slider-2',
                    min=2017,
                    max=2022, 
                    marks={
                    2017: '2017',                        
                    2018: '2018',                        
                    2019: '2019',
                    2020: '2020',
                    2021: '2021',
                    2022: '2022'},                    
                    
                    tooltip={"placement": "bottom", "always_visible": False},
                    step=1,
                    value=2017
                )
                ], width=4),                
                
                
                dbc.Col([
                html.H4("Influencers", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                        dcc.Dropdown(
                            id='tweets-dropdown',
                            placeholder="Select  BTC Compoonent",
                     options=[{'label': i, 'value': i}
                              for i in tweets_df["Username"].unique()],
                            style=dict(
                                width='100%',
                                verticalAlign="center",
                                justifyContent = 'center',
                                border= "1px solid black",
                                backgroundColor= "#f0f8ff",                                
                                fontSize=20,                                
                                fontFamliy = 'courier'),
                            value='MartyBent'),
                    
                ], width=4),              
            ], align='center'),                 
            
            html.Br(),
            
            dbc.Row([

                
                dbc.Col([
                html.H4("Correlation", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="correlation_plot_2")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),
                               
                
                dbc.Col([
                html.H4("Cointegration p-value", className="card-title", 
                        style={"fontFamily": "courier", "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="cointegration_plot_2")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),  
                
                
                dbc.Col([
                html.H4("Causality p-value", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="causality_plot_2")
                            ])
                            
                            
                        ],style={"background-color": ""}),style={"border": "1px solid black"}),
                ], width=4),
                
                dbc.Col([

                ], width=3),
            ], align='center'), 
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                html.H4("Positive vs Negative", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="poscos_bar_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6),  
                
                dbc.Col([
                html.H4("World Cloud", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="worldcloud_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=6), 
            ], align='center'),                 
                
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                html.H4("Sentiment Trend", className="card-title", style={"fontFamily": "courier",
                                                                      "textAlign":"center"}),
                    dbc.Card(
                        dbc.CardBody([
                            html.Div([
                                dcc.Graph(id="sentiment_trend_plot")  
                            ])
                            
                        ]),style={"border": "1px solid black"}),
                ], width=12),                 
            ], align='center'), 
            
            html.Br(),

        ],style={"border": "1px solid black","background-color": "#FDEEF4",'width':'73rem'})
    )]),style={"border": "2px solid black","background-color": "#FDEEF4",'width':'75rem'})
])  
    
    
    

###################################################     Callbacks         ########################################    

# Led & Slider Callback    
@app.callback(
    Output('my-LED-display-1', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_output(value):
    return str(value)

# Led 2 & Slider 2 Callback    
@app.callback(
    Output('my-LED-display-2', 'value'),
    Input('my-LED-display-slider-2', 'value')
)
def update_output(value):
    return str(value)


# this callback is use to toggle the "dash-bootstrap" class so you can see
# the effect of the custom stylesheets when running the example app.
@app.callback(Output("container", "className"), Input("toggle", "value"))
def toggle_classname(value):
    if 1 in value:
        return "dash-bootstrap"
    return ""


############################################## Home Page Callbacks #################################################
@app.callback(
    Output(component_id='mean_indicator_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value'),      
    Input(component_id='my-LED-display-slider-2', component_property='value')  
)

# Mean Indicator
def mean_indicator(selected_variable, selected_month, selected_year):

    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]
    df = df[df["year"] == selected_year]
    
    mean_indicator = df.value.mean()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(mean_indicator,2),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':',.0f'},         
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=80,paper_bgcolor = "#fffafa")

    fig.update_traces(number_font_family="courier")

    return fig


# Min Indicator
@app.callback(
    Output(component_id='min_indicator_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')  

)

# Mean Indicator
def min_indicator(selected_variable, selected_month, selected_year):
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]
    df = df[df["year"] == selected_year]

    min_indicator = df.value.min()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(min_indicator,2),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':',.0f'},         
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=80,paper_bgcolor = "#fffafa")
    fig.update_traces(number_font_family="courier")

    return fig

@app.callback(
    Output(component_id='max_indicator_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value') 

)

# Max Indicator
def max_indicator(selected_variable, selected_month, selected_year):
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]
    df = df[df["year"] == selected_year]
    max_indicator = df.value.max()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(max_indicator,2),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':',.0f'},   
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=80,paper_bgcolor = "#fffafa")

    fig.update_traces(number_font_family="courier")

    return fig

@app.callback(
    Output(component_id='std_indicator_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value') 

)
# Standard Deviation Indicator
def std_indicator(selected_variable, selected_month, selected_year):
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]
    df = df[df["year"] == selected_year]  


    std_indicator = df.value.std()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(std_indicator,2),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':',.0f'},         
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=80,paper_bgcolor = "#fffafa")

    fig.update_traces(number_font_family="courier")

    return fig


@app.callback(
    Output(component_id='areachart_trend_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)
# Area Chart Trend
def areachart_trend(selected_variable):

    df = df_melted[df_melted["variable"] == selected_variable]
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.Date, 
                             y=df.value, 
        name="Close",
        line=dict(
            color='#9c7c38',         
            width=2)
    ))
    
    # Add shape regions
    fig.add_vrect(
        x0="2021-02-28", x1="2021-03-31", annotation_text="COVID-19", annotation_font_size=18,
        fillcolor="#b2beb5", opacity=0.4,
        layer="above", line_width=0, 
    ),

    fig.add_vrect(
        x0="2022-02-20", x1="2022-03-22", annotation_text="Ukraine Crisis", annotation_font_size=18,
        fillcolor="#b2beb5", opacity=0.4,
        layer="above", line_width=0,
    )    
    



    fig.update_xaxes(title_text = '')
    fig.update_yaxes(title_text = '')
    fig.update_layout(template= 'gridon',  
                            font_family="Courier", # Set Font style
                            font_size=14, # Set Font size) # legend false  
                            height=275,
                            margin=dict(l=30, r=20, t=10, b=0))
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )    
    
        # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)   
    return fig


@app.callback(
    Output(component_id='distribution_hist_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')      

)

# Histogram
def histogram(selected_variable, selected_month, selected_year):
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]
    df = df[df["year"] == selected_year]  
  
    
    close_histogram = go.Figure()

    close_histogram.add_trace(go.Histogram(x=df.value, marker_color='#777696'))

    close_histogram.update_layout(template= 'gridon', 
                                  height = 150,
                                  font_family="Courier", # Set Font style
                                  font_size=14, # Set Font size) # legend false                               
                                  margin=dict(l=50, r=0, t=0, b=20))
    
    
    close_histogram.update_traces(histnorm="density")

    return close_histogram




# Boxplot Yearly
@app.callback(
    Output(component_id='distribution_bp_plot_y', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-1', component_property='value')    

)


def boxplot_y(selected_variable, selected_month):
    
    colorscale = ['#997a8d']
    
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["month"] == selected_month]    
    

    box_plot_fig = px.box(df, 
                          x="year", 
                          y="value",                
                          notched=True, # used notched shape
                          title="",
                          template='seaborn',
                          color_discrete_sequence=colorscale,
                          points="all")
                        
                


    box_plot_fig.update_yaxes(title_text = '')
    box_plot_fig.update_layout(template= 'gridon',  
                               height = 275,
                               font_family="Courier", # Set Font style
                               font_size=14, # Set Font size) # legend false                            
                               margin=dict(l=50, r=0, t=20, b=40))

    return box_plot_fig 

# Boxplot Monthly
@app.callback(
    Output(component_id='distribution_bp_plot_m', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')     

)


def boxplot_m(selected_variable,selected_year):
    
    colorscale = ["#666699"]
    
    df = df_melted[df_melted["variable"] == selected_variable]
    df = df[df["year"] == selected_year]    

    box_plot_fig = px.box(df, 
                          x="month", 
                          y="value",                
                          notched=True, # used notched shape
                          title="",
                          template='seaborn',
                          color_discrete_sequence=colorscale,
                          points="all")


    box_plot_fig.update_yaxes(title_text = '')
    box_plot_fig.update_layout(template= 'gridon', 
                               height = 275,
                               font_family="Courier", # Set Font style
                               font_size=14, # Set Font size) # legend false                            
                               margin=dict(l=50, r=0, t=20, b=40))

    return box_plot_fig 



############################################## Page 1 Callbacks #################################################
# Correlation
@app.callback(
    Output(component_id='correlation_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date')    

)

def correlation(selected_variable, start_date, end_date):
    

    
    
    
    df = df_melted[(df_melted["variable"] == "Close") |
                   (df_melted["variable"]== selected_variable)]


    df = df.loc[df["Date"].between(*pd.to_datetime([start_date, end_date]))]
    
    
    
    df = df.iloc[:,0:3]

    
    df = df.pivot(index='Date', columns='variable', values='value')
    
    corr_test = df.iloc[:,0:2].corr()
    

    corr_test = corr_test["Close"].iloc[1:2].values[0]


    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(corr_test,6),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test


# Cointegration
@app.callback(
    Output(component_id='cointegration_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date')    


)


def cointegration(selected_variable, start_date, end_date):
    
    df = df_melted[(df_melted["variable"] == "Close") |
                   (df_melted["variable"]== selected_variable)]


    df = df.loc[df["Date"].between(*pd.to_datetime([start_date, end_date]))]
    
    
    
    df = df.iloc[:,0:3]
    
    df = df.pivot(index='Date', columns='variable', values='value')
    
    score,pvalue,_=coint(df.iloc[:, 0],df.iloc[:, 1])

    pvalue = pvalue


    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(pvalue,6),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test



# Causality
@app.callback(
    Output(component_id='causality_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date')   

)
   
def causality(selected_component, start_date, end_date):
    
    
    df = df_melted[(df_melted["variable"] == "Close") |
                   (df_melted["variable"]== selected_component)]


    df = df.loc[df["Date"].between(*pd.to_datetime([start_date, end_date]))]  
    
    df = df.iloc[:,0:3]
    
    df = df.pivot(index='Date', columns='variable', values='value')
    
    data = df
    
  
    variables=data.columns  
    matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in matrix.columns:
        for row in matrix.index:
            test_result = grangercausalitytests(data[[row, col]], maxlag=4, verbose=False)            
            p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(4)]            
            min_p_value = np.min(p_values)
            matrix.loc[row, col] = min_p_value
    matrix.columns = [var + '_x' for var in variables]
    matrix.index = [var + '_y' for var in variables]
    
    matrix = matrix["Close_x"].to_frame()
    
    matrix = matrix.reset_index()

    matrix = matrix.rename(columns={ "index": "Components",
                                     "Close_x": "p-value"}).sort_values("p-value", ascending=True) 
    
    
    
    matrix = matrix.iloc[0:1,1:2]["p-value"].values[0]
    

    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(matrix,4),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test





# Scatter Matrix
@app.callback(
    Output(component_id='correlation_matrix_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value'),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date')     

)



def scatter_matrix(selected_variable,  start_date, end_date):
    
    df = df_melted[
        (df_melted["variable"]== selected_variable)|      
        (df_melted["variable"] == "Close")
                   ]

    df = df.loc[df["Date"].between(*pd.to_datetime([start_date, end_date]))]  
    
    
    
    df = df.iloc[:,0:3]

    
    df = df.pivot(index='Date', columns='variable', values='value')
    

    fig = go.Figure()

    fig.add_trace(go.Splom(dimensions=[
        dict(label=selected_variable, values = df[selected_variable]),       
        dict(label='Close', values = df['Close'])],
                           marker = dict(color = "#b0c4de",
                                         opacity = 0.8,
                                         size=15,
                                         line_color = '#666699',line_width=3),
                           showupperhalf = True,
                           showlowerhalf = True,
                            diagonal_visible=False                            
                           ),
                 )
    fig.update_layout(template= 'gridon',  
                                    height=325,

                                    font_family="Courier", # Set Font style
                                    font_size=14, # Set Font size) # legend false 
                                    margin=dict(l=70, r=0, t=30, b=40))
    return fig


    


# Standardised Trend
@app.callback(
    Output(component_id='standardised_trend_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)
def standardised_trend(selected_variable):

    #df = df_melted[df_melted["month"] == selected_month]
    #df = df_melted[df_melted["year"] == selected_year]

    
    df_close = df_melted.query("variable == 'Close'") 

    cols_1 = ["value"]
    corr_1_scaled = StandardScaler()
    df_close[cols_1] = corr_1_scaled.fit_transform(df_close[cols_1])
    

    df = df_melted[df_melted["variable"] == selected_variable]      
    
    cols_2 = ["value"]
    corr_2_scaled = StandardScaler()
    df[cols_2] = corr_2_scaled.fit_transform(df[cols_2])   
    
    block_fig = go.Figure()

    block_fig.add_trace(go.Scatter(
        x=df_close.Date, 
        y=df_close.value, 
        name="Close",
        line=dict(
            color='#9c7c38',
            width=2)
    ))

    block_fig.add_trace(go.Scatter(
        x=df.Date,                         
        y=df.value,                          
        name=selected_variable,
        line=dict(
            color="#989898",
            width=2)
    ))        

    block_fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    block_fig.update_layout(template= 'gridon',  
                            height=325,
                            font_family="Courier", # Set Font style
                            font_size=14, # Set Font size) # legend false                         
                            margin=dict(l=40, r=0, t=30, b=40))
    
    block_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )    
    
        # Add Spikes
    block_fig.update_xaxes(showspikes=True)
    block_fig.update_yaxes(showspikes=True)

    return block_fig




# Causality Table
@app.callback(
    Output(component_id='causality_table', component_property='figure'),  
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date')    

)

def causality_results(start_date, end_date):
    
    df_2 = df
    
    df_2["Date"] = pd.to_datetime(df_2["Date"])
    df_2["month"] = df_2["Date"].dt.month
    df_2["year"] = df_2["Date"].dt.year
    

    df_2 = df_2.loc[df_2["Date"].between(*pd.to_datetime([start_date, end_date]))]  
    
    data_filtered = df_2.set_index("Date")
    
    data_filtered = data_filtered.drop(columns=["month", "year", "returns", "log_returns"])
    
    # Calculate the Granger Causality
    variables=data_filtered.columns  
    matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in matrix.columns:
        for row in matrix.index:
            test_result = grangercausalitytests(data_filtered [[row, col]], maxlag=4, verbose=False)            
            p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(4)]            
            min_p_value = np.min(p_values)
            matrix.loc[row, col] = min_p_value


    matrix.columns = [var  for var in variables]
    matrix.index = [var for var in variables]

    # Select only the p-value for the Close Price
    matrix = matrix["Close"].to_frame()
    
    matrix = matrix[matrix["Close"] <= 0.05]
    
    matrix= matrix.reset_index()
    matrix = matrix.rename(columns={ "index": "Components",
                                     "Close": "p-value"}).sort_values("p-value", ascending=True)
    
    
    colorscale = [[0, '#666699'],[.5, '#dcdcdc'],[1, '#ffffff']]    
    fig = ff.create_table(matrix, colorscale=colorscale)

    
    return fig
    



############################################## Page 2 Callbacks #################################################

# AD Fuller
@app.callback(
    Output(component_id='ad_fuller_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)


def ad_fuller(selected_variable):

    df = df_melted[df_melted["variable"] == selected_variable]
    
    value = df.value.values
    result = adfuller(value)

    adf = result[0]

    pvalue = result[1]

    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(pvalue,5),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60)

    fig.update_traces(number_font_family="courier")

    return fig



@app.callback(
    Output(component_id='volatility_trend_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)
# Volatility Trend
def volatility_trend(selected_variable):
    
    #df = df_melted[df_melted["month"] == selected_month]
    #df = df_melted[df_melted["year"] == selected_year]
    df = df_melted[df_melted["variable"] == selected_variable] 

    
    df = df.set_index('Date')
    
    returns = 100 * df.value.pct_change().dropna()

    returns = returns.reset_index()

    # Variables to plug in the Confidence Interval Formula
    x = returns.value
    mean_pred = returns.value.mean() # average 
    alpha = 1.960 # Confidence Interval chosen was at 95%. Thus, based on the statistical table 1.960
    sqrt_n = np.sqrt(len(returns.value)) # Square roots of the sample size

        # Set Upper and Lower bounds for the Predicted Value
    returns['lower_pred'] = x - alpha * (mean_pred / sqrt_n)

    returns['upper_pred'] = x + alpha * (mean_pred / sqrt_n)

        # Create Figure
    fig = go.Figure()

        # Add traces observed values 
    fig.add_trace(go.Scatter(x=returns.Date, y=returns.value,
                            mode='lines',
                            line=dict(color='#666699', width=1),
                            name='Returns'))


        # Use update_layout in order to define few configuration such as figure height and width, title, etc
    fig.update_layout(
            title={
                'text': '', # Subplot main title
                'y':0.99, # Set main title y-axis position
                'x':0.5, # Set main title x-axis position
                'xanchor': 'center', # xachor position
                'yanchor': 'top'}, # yachor position 
            showlegend=False,
            font_family="Courier", # Set Font style
            font_size=14) # Set Font size) # legend false 

        # Update Styling
    fig.update_layout(hovermode="x", 
                        template =  'gridon', 
                         height = 300,
                        margin=dict(l=40, r=0, t=20, b=20))
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )    
    


        # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
  
    return fig




# PACF / ACF 
@app.callback(
    Output(component_id='pacf_acf_plot', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)
def acf_pacf(selected_variable):
    
    #df = df_melted[df_melted["month"] == selected_month]
    #df = df_melted[df_melted["year"] == selected_year]
    df = df_melted[df_melted["variable"] == selected_variable]
    
    df = df.set_index('Date')
    returns = 100 * df.value.pct_change().dropna()

    returns = returns.reset_index() 
    
    # Create subplots figure
    autocorrelation_plots = make_subplots(rows=2, cols=1, subplot_titles=("Partial Autocorrelation", "Autocorrelation"))

    # Partial Autocorrelation Plot (PACF)
    corr_array = pacf((returns.value[-1825:]).dropna(), alpha=0.05) 
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    # Autocorrelation Plot (ACF)
    corr_array_acf = acf((returns.value[-1825:]).dropna(), alpha=0.05) 
    lower_acf = corr_array_acf[1][:,0] - corr_array_acf[0]
    upper_acf = corr_array_acf[1][:,1] - corr_array_acf[0]


    # Partial Autocorrelation Plot
    [autocorrelation_plots.add_trace(go.Scatter(x=(x,x), 
                                  y=(0,corr_array[0][x]), 
                                  mode='lines',
                                  line_color='#3f3f3f',
                                  name = 'PACF'),
                       row=1,col=1)


        for x in range(len(corr_array[0]))]


    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                                 y=corr_array[0], 
                                 mode='markers', 
                                 marker_color='#090059',
                                 marker_size=12,
                                 name = 'PACF'),
                      row=1,col=1)

    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                                 y=upper_y, 
                                 mode='lines', 
                                 line_color='rgba(255,255,255,0)',
                                 name = 'Upper Bound'),
                     row=1,col=1)


    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                                 y=lower_y, 
                                 mode='lines',
                                 fillcolor='rgba(32, 146, 230,0.3)',
                                 fill='tonexty', 
                                 line_color='rgba(255,255,255,0)',
                                 name = 'Lower Bound'),
                     row=1,col=1)





    [autocorrelation_plots.add_trace(go.Scatter(x=(x,x), 
                                  y=(0,corr_array_acf[0][x]), 
                                  mode='lines',
                                  line_color='#3f3f3f',
                                  name = 'ACF'), 
                      row =2, col=1)


        for x in range(len(corr_array_acf[0]))]


    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                                 y=corr_array_acf[0], 
                                 mode='markers', 
                                 marker_color='#090059',
                                 marker_size=12,
                                 name = 'ACF'),
                      row =2, col=1)


    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                                 y=upper_acf, 
                                 mode='lines', 
                                 line_color='rgba(255,255,255,0)',
                                 name = 'Upper Bound'),
                      row =2, col=1)

    autocorrelation_plots.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                                 y=lower_acf, 
                                 mode='lines',
                                 fillcolor='rgba(32, 146, 230,0.3)',
                                 fill='tonexty', 
                                 line_color='rgba(255,255,255,0)',
                                 name = 'Lower Bound'),
                      row=2,col=1)




    # Update Figures
    autocorrelation_plots.update_traces(showlegend=False)
    autocorrelation_plots.update_xaxes(range=[-1,35],showspikes=True)
    autocorrelation_plots.update_yaxes(zerolinecolor='#000000',showspikes=True)

    # title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    autocorrelation_plots.update_layout(template = 'gridon', 
                            margin=dict(l=0, r=0, t=30, b=20),
                            height = 425,
                            showlegend=False,
                            font_family="Courier", # Set Font style
                            font_size=14) # Set Font size) # legend false 

    autocorrelation_plots.update_annotations(font=dict(family="Courier",size=24))

    # Return the subplots
    return autocorrelation_plots

    

    
    
# Seasonality Components
@app.callback(
    Output(component_id='seasonal_plots', component_property='figure'),  
    Input(component_id='btc-components-dropdown', component_property='value')

)
def seasonality_components(selected_variable):
    
    #df = df_melted[df_melted["month"] == selected_month]
    #df = df_melted[df_melted["year"] == selected_year]
    df = df_melted[df_melted["variable"] == selected_variable]
    
    df = df.set_index('Date')
    returns = 100 * df.value.pct_change().dropna()

    #returns = returns.reset_index()    

    result = seasonal_decompose(
                returns, model='additive', filt=None, period=365,
                two_sided=True, extrapolate_trend=0)
    seasonal = make_subplots(
                rows=4, cols=1,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
    seasonal.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name = "Observed", marker_color='#91a3b0'),
                    row=1, col=1
                )

    seasonal.add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name = "Trend", marker_color='#997a8d'),
                    row=2, col=1
                )

    seasonal.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name = "Seasonal", marker_color='#666699'),
                    row=3, col=1
                )

    seasonal.add_trace(
                go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name = "Residuals", marker_color='#757575'),
                    row=4, col=1
                )

    seasonal.update_layout(template = 'gridon', 
                            margin=dict(l=20, r=0, t=30, b=20),
                            height = 425,
                            showlegend=False,
                            font_family="Courier", # Set Font style
                            font_size=14) # Set Font size) # legend false 


    seasonal.update_annotations(font=dict(family="Courier",size=24))
    
    return seasonal



############################################## Page 3 Callbacks #################################################
 
############################################## Page 1 Callbacks #################################################
# Correlation 2
@app.callback(
    Output(component_id='correlation_plot_2', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')    

)

def correlation_2(selected_influencer,  selected_year):
    
    close = df[["Date","returns"]]
    
    sentiment = tweets_df[["Datetime", "sentiment", "Username"]].sort_values("Datetime")
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d').dt.tz_localize(None)
    
    sentiment["Datetime"]  = sentiment["Datetime"].dt.date
    
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d')    
    sentiment["month"]  = sentiment["Datetime"].dt.month
    sentiment["year"]  = sentiment["Datetime"].dt.year
    
    
    
    sentiment=sentiment.groupby(["Datetime", "month", "year", "Username"], as_index=False)\
    .mean().sort_values("Datetime")

    sentiment = sentiment.set_index("Datetime")
    close = close.set_index("Date")
    sentiment = sentiment.interpolate()
    close = close["2016-07-08": "2022-06-30"]
    close = close.reset_index()
    sentiment = sentiment.reset_index()
    df_corr_sentiment  = pd.concat([sentiment, close], axis=1)
    df_corr_sentiment = df_corr_sentiment[["Datetime", "month", "year", "sentiment", "Username", "returns"]]
    df_corr_sentiment = df_corr_sentiment.set_index("Datetime")

    df_final = df_corr_sentiment
    
    df_filtered = df_final[df_final["Username"] == selected_influencer]
    
    df_filtered_2 = df_filtered[                 
                                (df_filtered["year"] == selected_year)]
    
    
    
    df_corr = df_filtered_2[["returns", "sentiment"]]

    
    corr_test = df_corr.corr()
    

    df_corr_sentiment = corr_test["returns"][1]

    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(df_corr_sentiment,6),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test


# Cointegration 2
@app.callback(
    Output(component_id='cointegration_plot_2', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')    

)

def cointegration_2(selected_influencer,  selected_year):
    
    close = df[["Date","returns"]]
    
    sentiment = tweets_df[["Datetime", "sentiment", "Username"]].sort_values("Datetime")
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d').dt.tz_localize(None)
    
    sentiment["Datetime"]  = sentiment["Datetime"].dt.date
    
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d')    
    sentiment["month"]  = sentiment["Datetime"].dt.month
    sentiment["year"]  = sentiment["Datetime"].dt.year
    
    
    
    sentiment=sentiment.groupby(["Datetime", "month", "year", "Username"], as_index=False)\
    .mean().sort_values("Datetime")

    sentiment = sentiment.set_index("Datetime")
    close = close.set_index("Date")
    sentiment = sentiment.interpolate()
    close = close["2016-07-08": "2022-06-30"]
    close = close.reset_index()
    sentiment = sentiment.reset_index()
    df_corr_sentiment  = pd.concat([sentiment, close], axis=1)
    df_corr_sentiment = df_corr_sentiment[["Datetime", "month", "year", "sentiment", "Username", "returns"]]
    df_corr_sentiment = df_corr_sentiment.set_index("Datetime")

    df_final = df_corr_sentiment
    
    df_filtered = df_final[df_final["Username"] == selected_influencer]
    
    df_filtered_2 = df_filtered[                  
                                (df_filtered["year"] == selected_year)]
    
    
    
    df_coint = df_filtered_2[["returns", "sentiment"]]

    
    score,pvalue,_=coint(df_coint["returns"],df_coint["sentiment"])

    pvalue = pvalue


    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(pvalue,6),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test




# Causality 2
@app.callback(
    Output(component_id='causality_plot_2', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')    

)

def causality_2(selected_influencer, selected_year):
    
    
    close = df[["Date","returns"]]
    
    sentiment = tweets_df[["Datetime", "sentiment", "Username"]].sort_values("Datetime")
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d').dt.tz_localize(None)
    
    sentiment["Datetime"]  = sentiment["Datetime"].dt.date
    
    sentiment["Datetime"] = pd.to_datetime(sentiment["Datetime"], format = '%Y/%m/%d')    
    sentiment["month"]  = sentiment["Datetime"].dt.month
    sentiment["year"]  = sentiment["Datetime"].dt.year
    
    
    
    sentiment=sentiment.groupby(["Datetime",  "year", "Username"], as_index=False)\
    .mean().sort_values("Datetime")

    sentiment = sentiment.set_index("Datetime")
    close = close.set_index("Date")
    sentiment = sentiment.interpolate()
    close = close["2016-07-08": "2022-06-30"]
    close = close.reset_index()
    sentiment = sentiment.reset_index()
    df_corr_sentiment  = pd.concat([sentiment, close], axis=1)
    df_corr_sentiment = df_corr_sentiment[["Datetime",  "year", "sentiment", "Username", "returns"]]
    df_corr_sentiment = df_corr_sentiment.set_index("Datetime")

    df_final = df_corr_sentiment
    
    df_filtered = df_final[df_final["Username"] == selected_influencer]
    
    df_filtered_2 = df_filtered[                   
                                (df_filtered["year"] == selected_year)]
    
    
    
    data = df_filtered_2[["returns", "sentiment"]]
  
    variables=data.columns  
    matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in matrix.columns:
        for row in matrix.index:
            test_result = grangercausalitytests(data[[row, col]], maxlag=4, verbose=False)            
            p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(4)]            
            min_p_value = np.min(p_values)
            matrix.loc[row, col] = min_p_value
    matrix.columns = [var + '_x' for var in variables]
    matrix.index = [var + '_y' for var in variables]
    
    matrix = matrix["returns_x"].to_frame()
    matrix = matrix.reset_index()

    matrix = matrix.rename(columns={ "index": "Influencer",
                                     "returns_x": "p-value"}).sort_values("p-value", ascending=True) 
    
    matrix = matrix.iloc[0:1,1:2]["p-value"].values[0]

    fig_corr_test = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = round(matrix,4),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.4f'},        
    title = {'text': ""}))

    fig_corr_test.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=60,paper_bgcolor = "#fffafa")

    fig_corr_test.update_traces(number_font_family="courier")

    return fig_corr_test

    
    
# World Cloud
@app.callback(
    Output(component_id='worldcloud_plot', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')       
    

)   


def world_cloud(selected_influencer, selected_year):
    
    df = tweets_df
        
    df = df[df["Username"] == selected_influencer]

    df = df[
           (df["year"] == selected_year)]
    
        
    
    df = df["Text"]
    
    df = df.astype(str)

    my_wordcloud = WordCloud(
        background_color='white',
    ).generate(' '.join(df))

    fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2')
    fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)

    return fig_wordcloud      


# Positive & Negative Bar Plot
@app.callback(
    Output(component_id='poscos_bar_plot', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value'),
    Input(component_id='my-LED-display-slider-2', component_property='value')       
   

)

def barplot_poscos(selected_influencer, selected_year):
    
    colorscale = ["#c41e3a", "#0072bb"]
    df = tweets_df
        
    df = df[df["Username"] == selected_influencer]

    df = df[
           (df["year"] == selected_year)]
    

    
    df["Sentiment"] = np.where(df.sentiment<=0, "Negative", "Positive")
    
    df = df[["sentiment", "Sentiment"]].groupby(["Sentiment"], as_index=False).count()\
    .sort_values("sentiment", ascending=False)

    fig = px.bar(df.sort_values("Sentiment",ascending=False), 
                 x="Sentiment", 
                 y= "sentiment",
                 color="Sentiment",
                 color_discrete_sequence=colorscale,
                 text ='sentiment')                      
                  
    
    fig.update_layout(template= 'gridon', 
                      showlegend=False, 
                      margin=dict(l=50, r=0, t=30, b=20), 
                      height=300)
    
    
    fig.update_xaxes(title_text="")
    
    return fig   


# Positive & Negative Bar Plot
@app.callback(
    Output(component_id='sentiment_trend_plot', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value')       

)

def sentiment_trend(selected_influencer):
    
    df_close = df_melted.query("variable == 'returns'") 

    cols_1 = ["value"]
    corr_1_scaled = MinMaxScaler()
    df_close[cols_1] = corr_1_scaled.fit_transform(df_close[cols_1])  
    
    df_close = df_close.set_index("Date")
    
    #df_close = df_close["2022-01-01":]
    
    df = tweets_df[tweets_df["Username"] == selected_influencer]
    
    cols_2 = ["sentiment"]
    corr_2_scaled = MinMaxScaler()
    df[cols_2] = corr_2_scaled.fit_transform(df[cols_2])      
    
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    
    df = df["2019-08-01":]
    
    trend = df["sentiment"].resample('D').mean().to_frame().reset_index()

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df_close.index, y=df_close.value, 
                             name="Returns",
                             marker_color="#9c7c38")) 
    
    fig.add_trace(go.Bar(x=df_close.index, 
                         y=trend.sentiment, 
                         name=selected_influencer,
                         marker_color="#989898",
                         width=3))
    
    
    fig.update_xaxes(rangeslider_visible=True)
    
    fig.update_yaxes(title_text="Standardised Data")
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font_family="Courier", # Set Font style
        font_size=14 # Set Font size) # legend false         
    ))
     
    
        # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)    

    fig.update_layout(template= 'gridon',  margin=dict(l=50, r=50, t=20, b=20), height=300)
    return fig     



@app.callback(
    Output(component_id='min_year_plot', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value')     

)

# Min Year Indicator
def min_year(selected_influencer):


    
    df = tweets_df.copy()
    
    df = df[df["Username"] == selected_influencer]
    
    df["Datetime"] = pd.to_datetime(df["Datetime"])
        
    df["year"] = df["Datetime"].dt.year
    
    df["year"] = df["year"].astype(int)
    
    min_year = df.year.min()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = min_year,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.0f'},         
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=45,paper_bgcolor = "#fffafa")

    fig.update_traces(number_font_family="courier")

    return fig

    
    
  
@app.callback(
    Output(component_id='max_year_plot', component_property='figure'),  
    Input(component_id='tweets-dropdown', component_property='value')     

)  
# Max Year Indicator
def min_year(selected_influencer):

    df = tweets_df.copy()
    
    df = df[df["Username"] == selected_influencer]

        
    df["Datetime"] = pd.to_datetime(df["Datetime"])
        
    df["year"] = df["Datetime"].dt.year
    
    df["year"] = df["year"].astype(int)    
    
    max_year = df.year.max()


    fig = go.Figure(go.Indicator(
    mode = "number",
       # gauge = {'shape': "bullet"},
    value = max_year,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    number = {'valueformat':'.0f'},          
    title = {'text': ""}))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          height=45,paper_bgcolor = "#fffafa")
    
    

    fig.update_traces(number_font_family="courier")

    return fig    


############################################################## End ###################################################

if __name__ == "__main__":
    app.run_server(debug=False, port="8069")
