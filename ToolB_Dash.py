import numpy as np
import re ; import string
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
import base64 ; import datetime


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = 'Clariti TooLB'
# the model google universal-sentence-encoder can be downloaded from this link
# if the model as be downloaded already you dont have to download it again
#"https://tfhub.dev/google/universal-sentence-encoder/4"
module_url = './tfhub_modules/063d866c06683311b44b4992fd46003be952409c'

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
 
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
nlp = spacy.load('en', disable=['parser', 'ner'])

def clean(aa):
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


header1= dbc.Card([ dbc.Row([dbc.Col([html.H2('RFP Capability Check')],width={"size": 6, "offset": 3}, style={'text-align':'left','margin-top': 30}),
                             dbc.Col([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width':'100px', 'height':'100px'})], width=1, style={'align-items':'right'})
                             ]) 
                        ],style={'backgroundColor':'#124654' })

        #  html.Div(["Input Capabilities: ",
        #       dcc.Input(id='capabilities',value='Ability to apply for a new address', type='text',debounce=True,
        #                 style={'width': '60%', 'height': 120,'verticalAlign':'top'})
        #                 ])

app.layout = html.Div(style={'background-color':'#124654' },children=[header1,
         html.Br(),
         html.H5("Use '--' to seperate each entry of requirement",style={'color': '#2274A5','margin-left': '15%'}),
         html.Div([ dbc.Row([ dbc.Col([ "Input Requirements:"],width={"size": 1.1} , style={'align-items':'left','margin-left': '2.8%' } ),
                             dbc.Col([ dcc.Textarea(id='capabilities',value='--Ability to apply for a new address',n_blur=1,
                                   style={'width': '60%', 'height': 150,'verticalAlign':'top','margin-left': '0.7%'})
                                   ])
                            ])
                  ])
         ,
        html.Br(),

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
@app.callback(Output('info_text', 'children'),Output("update-table","children"),[Input("id_button","n_clicks")
                                       ,State("domain_id","value"),State("accuracy_id","value"),State("capabilities","value") ])

def gg(id_button,domain_id,accuracy_id,capabilities):
    queries =str(capabilities)
    inputted_req =  queries.lstrip('--').split('--')
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

#         bb=bb.set_index('index')\
#                     .sort_values('nos',ascending=True).reset_index(drop=True)
# bb.index =np.arange(1,len(bb)+1)
# bb.reset_index()

        dis = pd.DataFrame({'Index':range(1,len(inputted_req)+1),'Requirement':inputted_req,'Nos of Capabilities':count}).set_index('Index')\
                                                                                                                    .sort_values('Nos of Capabilities',ascending=False)\
                                                                                                                    .reset_index(drop=True)
        dis.index = np.arange(1,len(dis)+1)
        dis = dis.reset_index()
        count_1=0
        for i,j in list_of_capa:
            if len(i)>0:
                count_1+=1
        perc = round((count_1/len(list_of_capa)*100),2)


        return f'We are able to cover {perc}% of the RFP',dbc.Col(dbc.Table.from_dataframe(dis, striped=True, bordered=True, hover=True),width=5)


if __name__ == '__main__':
    app.run_server(debug=True,port=8088)