import json

from LLM.base_openai import *
import pandas as pd
import argparse
from tqdm import tqdm
def main(args):
    data=pd.read_excel(args.dataset_path)
    index=0
    check_bias=data['Bias Name']
    record_scripts=data['record_script']
    record_formats=data['record_format']
    logs=[]
    res=[]
    costs=[]
    curcase=[]
    LLM=OpenAI_chat('./Key_GPT_0.txt','https://api.openai.com/v1')
    for index in tqdm(range(0,len(check_bias))):
        record_log = 'The scene script is'+record_scripts[index]+'\n\n'+'The scene log is:'
        json_records=json.loads(record_formats[index])
        for record in json_records:
            origin=record['origin']
            target=record['target']
            trans_way=record['trans_way']
            content=record['content']
            if trans_way=='self-receiving':
                continue
            elif trans_way=='unicast':
                record_log+=f'{origin} to {target} said:{content}\n'
            elif trans_way=='broadcast':
                record_log+=f'{origin} to all said:{content}\n'
            elif trans_way=='multicast':
                record_log+=f'{origin} to {target} said:{content}\n'
        logs.append(record_log)
        system = {'role': 'system',
                  'content': f'You are an expert in cognitive bias detection, and now you have a very long scene script and scene dialogue. Please analyze whether there is {check_bias[index]} in this script'}
        user = {'role': 'user',
                'content': f'{record_log}\n\nPlease check whether there is {check_bias[index]} on the Subject in the scene log. If there is, analyze the reasons and details. The output format is JSON format, and the output content format is: {{"eval":"yes or not","reason":"xxx" }}Please note that the output content is only in JSON format, and do not output other content.'}
        request = [system, user]
        response = LLM.get_LLM_message(request,args.used_model)
        print(response)
        res.append(response)
        curcase.append(record_scripts[index])

        pd.DataFrame({"case":curcase,'res':res}).to_excel('./res/DB_gpt4_res.xlsx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--used_model", default="gpt-4-turbo", type=str, help='model')
    parser.add_argument("--dataset_path", default='./Data/dynamic_data/DB_gpt4_turbo_log.xlsx', type=str, help='dataset path')
    args = parser.parse_args()
    main(args)
