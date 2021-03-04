import numpy as np
import re , io ; import string
import pandas as pd
import nltk
from nltk.corpus import wordnet,stopwords
import scipy.spatial
import os ; import pickle
import tensorflow as tf
import tensorflow_hub as hub
import dash_table ; import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import spacy
import en_core_web_sm as en
import base64 ; import datetime


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = 'Clariti Requirement'
# the model google universal-sentence-encoder can be downloaded from this link
# if the model as be downloaded already you dont have to download it again
#"https://tfhub.dev/google/universal-sentence-encoder/4"

module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'

# Import the Universal Sentence Encoder's TF Hub module
model = hub.load(module_url)

all_files = os.listdir("app2_file")
all_files = filter(lambda x: x != '.DS_Store', all_files)
all_files = sorted(all_files ,key = lambda date: datetime.datetime.strptime(date, '%Y-%b-%d_%X'),reverse=True)
file_path = os.path.join(os.getcwd(),'app2_file',all_files[0])
print('\n',all_files,'\n')

#print out with file is been used the file name is date it was created
print(file_path,'\n')

PIK=f'{file_path}/All_Domains.plk' 

# load saved train universal encoded model 
pickle_domain= []
with open(PIK, "rb") as f:
    for _ in range(pickle.load(f)):
        pickle_domain.append(pickle.load(f))

# load safe csv files
pickle_raw=[]
for x in ('All_Capabilities','Finance_domain', 'Inspections_domain', 'Permits_domain',
              'Citizen_domain','Planning_domain','Licenses_domain','Code_domain'):
    x= pd.read_csv(f'{file_path}/{x}.csv')['Capabilties'].tolist()
    pickle_raw.append(x)

domain_id = ['All Domain','Finance', 'Inspections', 'Permits', 'Citizen Request',
             'Planning and Zoning', 'Licenses', 'Code Enforcement']


stop_words = set(stopwords.words('english'))

def remove_stopwords(tokenized_text): 
    text = tokenized_text.split(' ')
    text = " ".join([word for word in text if word not in stop_words])
    return text
 

nlp = en.load(disable=['parser', 'ner'])

def clean(aa):
    aa=str(aa)
    aa = aa.lower()
    aa = remove_stopwords(aa)
    aa = re.sub('[%s]' % re.escape('/-()'), ' ', aa)
    table = str.maketrans('', '', string.punctuation)
    aa = aa.translate(table)
    aa= re.sub(r"\s+", ' ', aa)
    aa = aa.strip()
    aa = nlp(aa)
    aa = " ".join([token.lemma_ for token in aa])
    aa = re.sub(r'\b(\w+)( \1\b)+', r'\1', aa)
    return aa


image_filename = './Clariti_logo_hero.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


header1= dbc.Card([ dbc.Row([dbc.Col([html.H2('RFP Capability Check')],width={"size": 6, "offset": 3}, style={'text-align':'left'}),
                             dbc.Col([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width':'100px', 'height':'100px'})], width=1, style={'align-items':'right'})
                             ]) 
                        ],style={'backgroundColor':'#124654' })


def parse_contents(contents, filename): 
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),names=['capa'])
    if df.shape[1] ==1:
        df.columns=['capa']
    elif df.shape[1] ==2:
        df.columns=['Index','capa']
    return filename,df

app.layout = html.Div(style={'background-color':'#124654' },children=[header1,
         html.Br(),
         
         html.Div([ dbc.Row([ dbc.Col([ "Upload CSV file: "],width={"size": 1} , style={'align-items':'left','margin-left': '2%','margin-top': 25} ),
                             dbc.Col([ dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
                                        style={
                                            'lineHeight': '60px',
                                            'borderWidth': '2px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=False
                                    )],width=2)
                            ])
                  ])
         ,html.Div(id='file_name',style={'color': '#2274A5','margin-left': '11%'})
         ,html.Br(),

        html.Div([ dbc.Row([ dbc.Col(['Choose Domain: '],width={"size": 1}, style={'align-items':'left','margin-left': '2%' }) , 
                            dbc.Col([dcc.Dropdown(id='domain_id',style={"Color": "black"},options=[{'label': v, 'value': k} for k, v in enumerate(domain_id)],
                                value=0,multi=False )],width=2) 
                           ],justify="start") 
                 ])
        
        ,
        
        html.Br(),
        
        html.Div([dbc.Row([
                      dbc.Col(["Accuracy Score: "],width={"size": 1},style={'align-items':'left','margin-left': '2%' }),
                      dbc.Col([dcc.Input(id='accuracy_id',type="number",min=0,max=100,debounce=True,value = 50,step=1)],width=2 ),
                      dbc.Col([html.Button('SEARCH', id='id_button',n_clicks = 0)])
                    ],justify="start")
             ]),
        html.Br(),
        html.Div([ html.H4(id='info_text',style={'color': '#2274A5','margin-left': '8.66666666667%' }) ,html.Div(id='update-table')
                        ]) 
        ])
         
@app.callback(Output('file_name', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        filename,df = parse_contents(list_of_contents, list_of_names)
        return html.Div([html.H5(f'Name of file Uploaded: {filename}') ])
    else: 
        return html.H5(['No file uploaded Yet'])




@app.callback(Output('info_text', 'children'),Output("update-table","children"),
                                        Input("id_button","n_clicks"),State('upload-data', 'contents')
                                       ,State("domain_id","value"),State("accuracy_id","value"),State('upload-data', 'filename'))

def gg(id_button,list_of_contents,domain_id,accuracy_id, list_of_names):
    filename,df = parse_contents(list_of_contents, list_of_names)
    inputted_req =  df['capa'].tolist()
    clean_queries = [clean(i) for i in inputted_req]
    query_embeddings = model(clean_queries)
    
    score =accuracy_id
    domain =domain_id
    list_of_capa =[]
    
    if id_button > 0:
        for query, query_embedding in zip(inputted_req, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], pickle_domain[domain], "cosine")[0]
            data = []
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            for idx, distance in results:
                x =(1-distance)*100
                if x >= score:
                    data.append(pickle_raw[domain][idx].strip())
            
            list_of_capa.append((data,len(data)))
        count = [j for i,j in list_of_capa]


        dis = pd.DataFrame({'Index':range(1,len(inputted_req)+1),'Requirement':inputted_req,'Nos of Capabilities':count}).set_index('Index')\
                                                                                                                    .sort_values('Nos of Capabilities',ascending=False)\
                                                                                                                    .reset_index(drop=True)
        dis.index = np.arange(1,len(dis)+1)
        dis = dis.reset_index()
        count_1=0
        for i,j in list_of_capa:
            if len(i)>0:
                count_1+=1
        perc = f'{(count_1/len(list_of_capa)*100):.2f}'


        return f'We are able to cover {perc}% of the RFP',dbc.Col(dbc.Table.from_dataframe(dis, striped=True, bordered=True, hover=True),width=5)


if __name__ == '__main__':
    app.run_server(debug=True)