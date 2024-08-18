import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from annoy import AnnoyIndex
import os
import torch
import json
import time
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer,BertModel
from random import randint
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

def get_openai_vec(text, engine="text-embedding-ada-002"):
    key_path = 'Openaikey/Key_GPT_0.txt'
    keyindex = randint(0, 3)
    if keyindex == 0:
        key_path = 'Openaikey/Key_GPT_0.txt'
    elif keyindex == 1:
        key_path = 'Openaikey/Key_GPT_1.txt'
    elif keyindex == 2:
        key_path = 'Openaikey/Key_GPT_2.txt'
    while True:
        try:
            openai.api_key = open(key_path, 'r').read()
            response = openai.Embedding.create(input=text, engine=engine)
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            print('api--' + key_path + ',did not work.--------- {}'.format(e))
            time.sleep(15)
            if '0' in key_path:
                key_path = key_path.replace('0', '1')
            elif '1' in key_path:
                key_path = key_path.replace('1', '2')
            elif '2' in key_path:
                key_path = key_path.replace('2', '0')

def get_embedding(text,model,tokenizer):
 
    inputs = tokenize_text(text,tokenizer)


    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    cls_embedding = embeddings[:, 0, :]

    mean_embedding = embeddings.mean(dim=1)


    combined_embedding = torch.cat((cls_embedding, mean_embedding), dim=1).squeeze().numpy()
    return  combined_embedding
def build_annoy_index(embeddings, n_trees=10):
    dim = len(next(iter(embeddings.values())))  
    annoy_index = AnnoyIndex(dim, 'angular')  


    for i, embedding in enumerate(embeddings.values()):
        annoy_index.add_item(i, embedding)

    annoy_index.build(n_trees) 
    return annoy_index


def find_similar_biases(scene_embedding, bias_embeddings, index, top_n):

    nearest_ids = index.get_nns_by_vector(scene_embedding, top_n)

    bias_names = list(bias_embeddings.keys())
    similar_biases = [bias_names[i] for i in nearest_ids]
    return similar_biases

def tokenize_text(text,tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return inputs

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
def main(threshold=10):
    model_name = "../bert-large-uncased" 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    file_path = '../Data/never_used_data/cognitive_bias.json'

    openai.api_base = "https://api.chatanywhere.com.cn/v1"

    with open(file_path,'r',encoding='utf-8') as file:
        biases=json.load(file)

    bias_embeddings = {name:  get_embedding(description,model,tokenizer) for name, description in biases.items()}
    annoy_index = build_annoy_index(bias_embeddings)

    scene_text = '''
    Scenario 7: As an independent film director, you observe that biopics are winning big at film festivals and awards ceremonies. You've always been passionate about science fiction and have a script that tackles complex ethical questions. Question: Do you set aside your script to work on a biopic instead?
    No, I would not set aside my script to work on a biopic. I am passionate about science fiction and have a script that tackles complex ethical questions. Working on a project that doesn't align with my interests would likely compromise the quality and authenticity of the film.
    '''
    test_case=pd.read_excel('../Data/check.xlsx')
    cnt1=0
    cnt2=0
    ans1=[]
    ans2=[]
    biases_list=[]
    for i in tqdm(range(len(test_case['biasname']))):
        curbias=test_case['biasname'][i]
        curcase=str(test_case['case'][i])+'\n'+str(test_case['GPT4'][i])
        biases_list.append(curbias)

        scene_embedding =  get_embedding(curcase,model,tokenizer)

        sorted_biases = find_similar_biases(scene_embedding, bias_embeddings, annoy_index,threshold)
        isshot1=False
        isshot2=False
        for bias in sorted_biases:
            if curbias==bias:
                cnt1+=1
                isshot1=True
        if isshot1:
            ans1.append(1)
        else:
            ans1.append(0)
        sorted_biases = sorted(biases.keys(), key=lambda bias: cosine_similarity(scene_embedding, bias_embeddings[bias]),
                                reverse=True)

        for bias in sorted_biases[0:threshold]:
            if curbias == bias:
                cnt2 += 1
                isshot2 = True
        if isshot2:
            ans2.append(1)
        else:
            ans2.append(0)
        res=pd.DataFrame({'biases_list':biases_list,'ans1':ans1,'ans2':ans2})
        res.to_excel('../Data/res.xlsx')
    print(f'method1:{ans1},acc:{cnt1/len(ans1)}')
    print(f'method2:{ans2},acc:{cnt2/len(ans2)}')
def leftonlycolum():
    df = pd.read_excel('../Data/cognitive bias.xlsx')
    # Removing duplicate entries in the 'biasname' column, keeping only the first occurrence
    df_unique = df.drop_duplicates(subset='biasname', keep='first')
    df_unique.to_excel('../Data/cognitive_bias_v2.xlsx')
def excel2json(file_path):
    df = pd.read_excel(file_path)
    biasname=df['biasname']
    description=df['description']
    gt_case=df['GT_case']
    for i in range(len(biasname)):
        description[i]=biasname[i]+" : "+description[i]+'\n Real case:'+str(gt_case[i])
    # Convert the dataframe to a dictionary with biasname as keys and descriptions as values
    bias_dict = df.set_index('biasname')['description'].to_dict()
    # Convert the dictionary to a JSON string
    json_output = json.dumps(bias_dict, indent=4)
    # Save the JSON string to a file
    output_file_path = '../Data/never_used_data/cognitive_bias.json'
    with open(output_file_path, 'w') as file:
        file.write(json_output)

def excel2csv(file_path):
    df = pd.read_excel(file_path)
    biasname=df['biasname']
    description=df['description']
    gt_case=df['GT_case']
    for i in range(len(biasname)):
        description[i]=biasname[i]+" : "+description[i]+'\n Real case:'+str(gt_case[i])
    res=pd.DataFrame({'biasname':biasname,'description':description})
    res.to_csv('../Data/cognitive_bias.csv')
#top10 0.24
#top20 0.43
#top30 0.59

if __name__ == '__main__':
    #leftonlycolum()
    # excel2json("../Data/congnitive bias_v1.xlsx")
    excel2csv("../Data/congnitive bias_v1.xlsx")
    # ths=[10,20,30,40,50]
    # for th in ths:
    #     main(th)
