import argparse
import json
from random import randint
import time
import openai
import os
import re
from LLM.azure_openai import Azure
from openai import OpenAI

def clean_json(text):
    extracted_text = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if extracted_text:
        result = '[' + extracted_text.group(1) + ']'
        return result


def clean_json_string(s):
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    pattern = r'\{.*\:\s*\{.*\:.*\}\}'
    match = re.search(pattern, s)
    if match:
        json_str = match.group()
        return json_str
    else:
        return ""


class BaseAgent:
    def __init__(self, model, key_path, log=None):
        self.system = ''
        self.user = ''
        self.used_model = model
        self.log = log
        self.key_path = key_path

        self.message = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user}
        ]

    def setMessage(self):
        self.message = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user}
        ]

    def response_LLMs(self, method='api', temperature=0.1):
        if method == 'api':
            while True:
                try:
                    api_key = open(self.key_path, 'r').read()
                    client =OpenAI(api_key=api_key,base_url='https://api.openai.com/v1')
                    response = client.chat.completions.create(
                        model=self.used_model,
                        messages=self.message,
                        temperature=0.2,
                        presence_penalty=1.0,
                        frequency_penalty=1.0,
                    
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    print('api--' + self.key_path + ',did not work.--------- {}'.format(e))
                    time.sleep(15)
                    if '0' in self.key_path:
                        self.key_path = self.key_path.replace('0', '1')
                    elif '1' in self.key_path:
                        self.key_path = self.key_path.replace('1', '2')
                    elif '2' in self.key_path:
                        self.key_path = self.key_path.replace('2', '0')




class checkBiasAgent(BaseAgent):
    def __init__(self, name=None, bias=None, knowledge=None, object=None, azure=None, model='gpt-4-turbo', WM_num=3,
                 key_path=None, log=None,personality=None):
        super().__init__(model, key_path, log)
        self.model = model
        self.personality=personality
        self.azure = azure
        self.name = name  
        self.bias = bias
        # system-prompt
        self.system = {"role": "system", "content": f'You are an expert in cognitive bias detection and your task is to detect the presence of {bias} in the text of a scene.'}

        self.LongMemory = []  
        self.shortMemory = []  
        self.knowledge = knowledge
        if object != None:
            self.object = '\n'.join([f"{key}: {value}" for key, value in object.items()])
        else:
            self.object = ""
        self.ChatLog = log 
        self.WM_num = 3  

        self.planning = ''  
        self.reflect = ''  
        self.check = False
        self.eval=''

    def receive_message(self, content):
        self.shortMemory.append(content)
        # self.ChatLog.write(content+'\n')

    def response_LLMs(self, message, method='api',temperature=0.1):
        if method == 'api':
   
            keyindex = randint(0, 3)
            if keyindex == 0:
                self.key_path = './Key_GPT_0.txt'
            elif keyindex == 1:
                self.key_path = './Key_GPT_1.txt'
            elif keyindex == 2:
                self.key_path = './Key_GPT_2.txt'
            
            while True:
                try:  
                    api_key = open('./Key_GPT_0.txt', 'r').read()
                    client =OpenAI(api_key=api_key,base_url='https://api.openai.com/v1')
                    response = client.chat.completions.create(
                        model=self.used_model,
                        messages=message,
                        temperature=0.2,
                        presence_penalty=1.0,
                        frequency_penalty=1.0,
                    
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    print('api--' + self.key_path + ',did not work.--------- {}'.format(e))
                    time.sleep(15)
                    if '0' in self.key_path:
                        self.key_path = self.key_path.replace('0', '1')
                    elif '1' in self.key_path:
                        self.key_path = self.key_path.replace('1', '2')
                    elif '2' in self.key_path:
                        self.key_path = self.key_path.replace('2', '0')
        # elif method == 'azure':
        #     response = self.azure.get_LLM_message(message, used_model=self.model,temperature=temperature)
        #     return response

    def scoring_one(self, record, name, bias, text,temperature=0):
        system = f'''your character is {self.personality}\n\n
        You will act as an evaluation system, responsible for objectively judging the performance of a debate. In this debate, there are two participants (referred to as Debater A and Debater B). You need to assess their performance based on the following criteria, and provide corresponding scores and brief reasons for your evaluation
        You need to rate the performance of each Agent according to the following six criteria.
        Argument Support:
        Quantitative measure: The degree to which an argument is supported by evidence or data. Scores may be given based on the quantity and quality of evidence provided.
        Rating range: 0 (no support) to 10 (well supported).

        Logical Consistency:
        Quantitative measure: The logical consistency of an argument throughout the debate. Evaluation criteria include the internal consistency of the arguments and the reasonableness of the conclusions.
        Rating range: 0 (completely inconsistent) to 10 (completely consistent).

        Refutation Effectiveness:
        Quantitative measure: The effectiveness of the Agent's refutation of the opponent's arguments. Consider the accuracy, relevance, and impact of your rebuttal.
        Scoring range: 0 (ineffective rebuttal) to 10 (highly effective rebuttal).

        Argument Completeness:
        Quantitative measure: The completeness of the argument in its presentation. Check whether all key elements are included, such as definitions, assumptions, arguments, and conclusions.
        Rating range: 0 (extremely incomplete) to 10 (very complete).

        Persuasiveness:
        Quantitative measure: The persuasiveness of an argument to an audience or judges. Criteria include the clarity, appeal and persuasiveness of the argument.
        Rating range: 0 (not at all convincing) to 10 (very convincing)

        Reasonability assessment of cognitive bias:
        Quantitative measurement: Evaluate whether the cognitive biases proposed in the original scene text are reasonable and have sufficient basis. This includes an appropriate analysis and reasonable explanation of whether the proposed deviation is based on the text content, as well as whether there is sufficient evidence to support the deviation indication.
        Rating range: 0 (deviation proposal has no reasonable basis) to 10 (deviation proposal is highly reasonable and supported by sufficient evidence).Please note that this indicator must ensure that the values for the two deviations are different.
        The description of the score is
        0 points: completely non compliant - completely not meeting relevant standards or requirements, lacking relevance or rationality.
        1~2 point: Very insufficient - largely not meeting relevant standards, lacking sufficient relevance or rationality.
        3~4 points: Insufficient - unable to meet relevant standards to a certain extent, with some relevance or rationality, but insufficient.
        5~6 points: General - basically meets relevant standards, has certain relevance and rationality, but there is still room for improvement.
        7~8 points: Good - Meets most relevant standards and demonstrates good correlation and rationality.
        9~10 points: Fully compliant - fully meets the relevant standards, demonstrating the highest degree of relevance and rationality.'''
        prompt = f'''
                Debate content:\n{record}\n\n
                The cognitive bias scenario is:\n{text}\n\n
                Scoring precautions:
                1. Please be careful not to lean towards the person who initiated the conversation first, and ensure that the rating is absolutely objective;
                2. Strictly follow the scoring criteria mentioned above and gradually analyze the debate content for scoring;
                3. Please try to avoid order bias as much as possible and avoid giving high scores to those who appear first.

                Please rate the performance of {name} in the following debate content based on the above criteria.\n
                '''
        output_format = f'''The output format is JSON.  Please note that you only need to output the content in JSON format and do not output any other content.
                for example:
                {{
                    "{name}": {{
                        "Argument Support": 0~10,
                        "Logical Consistency": 0~10,
                        "Refutation Effectiveness": 0~10,
                        "Argument Completeness": 0~10,
                        "Persuasiveness":0~10,
                        "Reasonability assessment of cognitive bias":0~10,
                        "cognitive bias name":"{bias}"
                    }}
                }}         
        '''
        requests = [{"role": "system", "content": system}, {"role": "user", "content": prompt + output_format}]
        response = self.response_LLMs(requests,temperature=temperature)
        response=response.lower()
        print(response)
        response = clean_json_string(response)
        # response = clean_json(response)

        time = 0
        while time < 2:
            try:
                response = json.loads(response)
                return response, True
            except:
                print('json error')
                response = self.response_LLMs(requests)
                time += 1
        return response, False

    def scoring(self, record, name1, name2, bias1, bias2, text):
        system = f'''
        You will serve as a scoring agent, responsible for giving structured scoring to a debate content. The debate content is provided by two different Agents, called Agent A and Agent B respectively. You need to rate the performance of each Agent according to the following six criteria.
        Scoring criteria
        Argument Support:
        Quantitative measure: The degree to which an argument is supported by evidence or data. Scores may be given based on the quantity and quality of evidence provided.
        Rating range: 0 (no support) to 10 (well supported).

        Logical Consistency:
        Quantitative measure: The logical consistency of an argument throughout the debate. Evaluation criteria include the internal consistency of the arguments and the reasonableness of the conclusions.
        Rating range: 0 (completely inconsistent) to 10 (completely consistent).

        Refutation Effectiveness:
        Quantitative measure: The effectiveness of the Agent's refutation of the opponent's arguments. Consider the accuracy, relevance, and impact of your rebuttal.
        Scoring range: 0 (ineffective rebuttal) to 10 (highly effective rebuttal).

        Argument Completeness:
        Quantitative measure: The completeness of the argument in its presentation. Check whether all key elements are included, such as definitions, assumptions, arguments, and conclusions.
        Rating range: 0 (extremely incomplete) to 10 (very complete).

        Persuasiveness:
        Quantitative measure: The persuasiveness of an argument to an audience or judges. Criteria include the clarity, appeal and persuasiveness of the argument.
        Rating range: 0 (not at all convincing) to 10 (very convincing)

        Reasonability assessment of cognitive bias:
        Quantitative measurement: Evaluate whether the cognitive biases proposed in the original scene text are reasonable and have sufficient basis. This includes an appropriate analysis and reasonable explanation of whether the proposed deviation is based on the text content, as well as whether there is sufficient evidence to support the deviation indication.
        Rating range: 0 (deviation proposal has no reasonable basis) to 10 (deviation proposal is highly reasonable and supported by sufficient evidence).Please note that this indicator must ensure that the values for the two deviations are different.
        The description of the score is
        0 points: completely non compliant - completely not meeting relevant standards or requirements, lacking relevance or rationality.
        1~2 point: Very insufficient - largely not meeting relevant standards, lacking sufficient relevance or rationality.
        3~4 points: Insufficient - unable to meet relevant standards to a certain extent, with some relevance or rationality, but insufficient.
        5~6 points: General - basically meets relevant standards, has certain relevance and rationality, but there is still room for improvement.
        7~8 points: Good - Meets most relevant standards and demonstrates good correlation and rationality.
        9~10 points: Fully compliant - fully meets the relevant standards, demonstrating the highest degree of relevance and rationality.
        Debate content:\n{record}\n\n
        The cognitive bias scenario in the debate between the two is:\n{text}\n\n
        Scoring precautions:
        1. Please be careful not to lean towards the person who initiated the conversation first, and ensure that the rating is absolutely objective;
        2. Strictly follow the scoring criteria mentioned above and gradually analyze the debate content for scoring;
        3. Please try to avoid order bias as much as possible and avoid giving high scores to those who appear first.
                '''
        prompt = f'''Please rate the performance of {name1} and {name2} in the following debate content based on the above criteria.The scores between the two are relative, and it is necessary to ensure the difference in scores between the two.\n
        '''
        output_format = f'''The output format is JSON.  Please note that you only need to output the content in JSON format and do not output any other content.
        for example:
        [
        {{
            "{name1}": {{
                "Argument Support": 8,
                "Logical Consistency": 2,
                "Refutation Effectiveness": 5,
                "Argument Completeness": 2
                "Persuasiveness":1,
                "Reasonability assessment of cognitive bias":10,
                "cognitive bias name":{bias1}
            }}
        }},
        {{
            "{name2}": {{
                "Argument Support": 2,
                "Logical Consistency": 1,
                "Refutation Effectiveness": 7,
                "Argument Completeness": 2
                "Persuasiveness":6,
                "Reasonability assessment of cognitive bias":1
                "cognitive bias name":{bias2}
            }}
        }}
        ]
'''
        requests = [{"role": "system", "content": system}, {"role": "user", "content": prompt + output_format}]
        response = self.response_LLMs(requests)
        print(response)
        # response=clean_json_string(response)
        response = clean_json(response)

        time = 0
        while time < 2:
            try:
                response = json.loads(response)
                return response, True
            except:
                print('json error')
                response = self.response_LLMs(requests)
                time += 1
        return response, False
    def checkself(self, text):

        prompt = f"The cognitive bias currently detected is {self.bias}. The detailed description of this bias is: <{self.knowledge}>. The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>. Please gradually consider whether this cognitive bias exists in the responses of characters in the scene text based on the characteristics of the bias and the attributes of concern. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
        request = [self.system, {"role": "user", "content": prompt}]
        # response = self.azure.get_LLM_message(request)
        response =self.response_LLMs(request)
        time=0
        while time<3:
            time+=1
            try:
                response_json = json.loads(response)
                if response_json['eval'] == 'yes':
                    response =self.response_LLMs(request)
                    response_json = json.loads(response)
                    print('The current detection bias is ' + self.bias + ' ' + response)
                else:
                    print('The current detection bias is ' + self.bias + ' ' + response)
                break
            except:
                print('try again!')
        return response_json
    # def checkself(self, text):
    #     input_prompt=f"The cognitive bias currently detected is {self.bias}.  The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>.The scene text includes case and response. Please gradually consider whether there is such cognitive bias in the responses in the scene text based on the characteristics of this cognitive bias and the attributes of concern. Please note not to conduct bias analysis on scene examples, only bias analysis on responses.  If there is a bias, it must be explained through strict analysis, otherwise it cannot be considered as biased."
    #     output_prompt="Output in JSON format, the output format is: " + '"{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Please note that yes represents clear indications of corresponding deviation, while no indicates no indication indicating no deviation or inability to clearly indicate the existence of corresponding deviationReason limit of 100 words'

    #     prompt = f"The cognitive bias currently detected is {self.bias}. The detailed description of this bias is: <{self.knowledge}>. The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>. Please gradually consider whether this cognitive bias exists in the responses of characters in the scene text based on the characteristics of the bias and the attributes of concern. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
    
    #     #prompt = f"Please analyze the provided scene text to determine if a specific cognitive bias is present. The cognitive bias to look for is '{self.bias}'. Pay special attention to the following attributes: '{self.object}'. Consider the following scene text: '{text}'. Based on the characteristics of the bias and the mentioned attributes, decide if the cognitive bias exists in the responses within the scene text. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'

    #     #prompt = f"The cognitive bias currently detected is {self.bias}. The detailed description of this bias is: <{self.knowledge}>. The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>. Please gradually consider whether this cognitive bias exists in the responses of characters in the scene text based on the characteristics of the bias and the attributes of concern. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
    #     #prompt=f"Please analyze the provided scene text to determine if a specific cognitive bias is present. The cognitive bias to look for is '{self.bias}'. A brief description of this bias is: '{self.knowledge}'. Pay special attention to the following attributes: '{self.object}'. Consider the following scene text: '{text}'. Based on the characteristics of the bias and the mentioned attributes, decide if the cognitive bias exists in the responses within the scene text. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
    #     request = [self.system, {"role": "user", "content": prompt}]
    #     response = self.azure.get_LLM_message(request)
    #     time=0
    #     while time<3:
    #         time+=1
    #         try:
    #             response_json = json.loads(response)
    #             if(response_json['eval']=='yes'):
    #                 request.append({"role": "user", "content": 'Here is your previous response:'+response+'\n\nPlease reflect on previous responses for problems and provide final cognitive bias test results.'+output_prompt})
    #                 response = self.azure.get_LLM_message(request)
    #                 response_json=json.loads(response)
    #             print('The current detection bias is ' + self.bias + ' ' + response)
    #             break
    #         except:
    #             print('try again!')
    #     return response_json
    # def checkself(self, text):
    #     input_prompt=f"The cognitive bias currently detected is {self.bias}.  The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>.The scene text includes case and response. Please gradually consider whether there is such cognitive bias in the responses in the scene text based on the characteristics of this cognitive bias and the attributes of concern. Please note not to conduct bias analysis on scene examples, only bias analysis on responses.  If there is a bias, it must be explained through strict analysis, otherwise it cannot be considered as biased."
    #     output_prompt="Output in JSON format, the output format is: " + '"{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Please note that yes represents clear indications of corresponding deviation, while no indicates no indication indicating no deviation or inability to clearly indicate the existence of corresponding deviationReason limit of 100 words'

    #     prompt = input_prompt+output_prompt
    #     #prompt = f"Please analyze the provided scene text to determine if a specific cognitive bias is present. The cognitive bias to look for is '{self.bias}'. Pay special attention to the following attributes: '{self.object}'. Consider the following scene text: '{text}'. Based on the characteristics of the bias and the mentioned attributes, decide if the cognitive bias exists in the responses within the scene text. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'

    #     #prompt = f"The cognitive bias currently detected is {self.bias}. The detailed description of this bias is: <{self.knowledge}>. The main attributes that need attention include: <{self.object}>. The scene text that needs to be detected is : <{text}>. Please gradually consider whether this cognitive bias exists in the responses of characters in the scene text based on the characteristics of the bias and the attributes of concern. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
    #     #prompt=f"Please analyze the provided scene text to determine if a specific cognitive bias is present. The cognitive bias to look for is '{self.bias}'. A brief description of this bias is: '{self.knowledge}'. Pay special attention to the following attributes: '{self.object}'. Consider the following scene text: '{text}'. Based on the characteristics of the bias and the mentioned attributes, decide if the cognitive bias exists in the responses within the scene text. Output in JSON format, the output format is: " + '{"eval":"yes or no","reason":"xxx"}Except for the content in {} above, other content is not allowed to be output.Reason limit of 100 words'
    #     request = [self.system, {"role": "user", "content": prompt}]
    #     response = self.azure.get_LLM_message(request)
    #     time=0
    #     while time<3:
    #         time+=1
    #         try:
    #             response_json = json.loads(response)
    #             if(response_json['eval']=='yes'):
    #                 request.append({"role": "user", "content": '\nHere are the results of your last test:'+response+'\n\nPlease gradually consider whether there were any issues with your last answer and provide you with the final test results'+output_prompt})
    #                 response = self.azure.get_LLM_message(request)
    #                 response_json=json.loads(response)
    #             print('The current detection bias is ' + self.bias + ' ' + response)
    #             break
    #         except:
    #             print('try again!')
    #     return response_json
    
    def chat(self, request):
        usedlongmemory = ""
        if len(self.LongMemory) != 0:
            for i in range(len(self.LongMemory)):
                curLongMemory = curLongMemory + self.LongMemory[i] + '\n' 
            prompt = [{"role": "system",
                       "content": 'As an expert who specializes in extracting key information from long-term memory, your task is to analyze a request and extract relevant information from long-term memory.'},
                      {"role": "user",
                       "content": f"The request content is: {request}\nThe long-term memory is: {curLongMemory}"}]
            usedlongmemory = self.response_LLMs(prompt)  
        curshortMemory = ""
        for i in range(len(self.shortMemory)):
            curshortMemory = curshortMemory + self.shortMemory[i] + '\n'
        prompt = [self.system, {'role': 'user',
                                "content": f'The long-term memory is: {usedlongmemory}\nThe memory of your short memory is: {curshortMemory}\n. Please combine the above memories to complete the following request: {request}\n' \
                                           f'Please try to be as smooth as possible in your answer and don’t be too rigid.'
                                }]
        response = self.response_LLMs(prompt)
        self.shortMemory.append('you yourself once said：' + response)
        # self.ChatLog.write('---' + self.role + '：\n' + response + '\n\n')
        return response

    def summary(self):
        curshortMemory = ""
        for i in range(len(self.shortMemory)):
            curshortMemory = curshortMemory + self.shortMemory[i] + '\n'
        prompt = {'role': 'user',
                  'content': f'The memory of your short memory is: {curshortMemory}.Please summarize the above content and pay attention to distinguish your own speech from that of others. The maximum length is 500 words.'}
        response = self.response_LLMs(prompt)
        self.LongMemory.append(response)

    def reflectaction(self):
        curshortMemory = ""
        for i in range(len(self.shortMemory)):
            curshortMemory = curshortMemory + self.shortMemory[i] + '\n'
        system = {'role': 'user',
                  'content': 'As a very rational person, you are good at discovering the logic in a conversation.'}
        prompt = [system, {"role": "user",
                           "content": f"The memory of your short memory is: {curshortMemory}.In order to better improve yourself, please identify the mistakes and give ways to improve them."}]
        self.reflect = self.response_LLMs(prompt)


class interpretationer(BaseAgent):
    def __init__(self, model, key_path, log):
        super().__init__(model, key_path, log)
        self.system = "你是一个场景内容提取器，你的任务是根据一个场景中特定内容，提取出内容为json格式"

    def interpret(self, rules):
        self.user = '请对以下规则逐条进行解释，解释后的形式为json,这个规则包括了场景类型，场景题目以及具体规则，在后面的内容中请结合场景题目和类型进行生成内容;' \
                    '发言的人指本条规则中发言的对象的名称;' \
                    '向谁发言指本条规则中要对谁发言，包括"全体"，"单播+那个对象的名称";' \
                    '行为prompt指本条规则中发言的人发言给与大模型的提示语;' \
                    '需要的技能指结合本次场景类型，本条规则中发言的人在本次场景中为了提高发言能力，而需要的技能，请从以下技能中选择：1.教学技能；2.辩论技能；3.学习技能；4.质询技能；5总结技能等等' \
                    '场景所需的人物中即从规则中提取中所有需要的人物，人物不要重复' \
                    'JSON应具有以下格式：\n["场景类型":"...","场景名称:"...","场景所需人物":[{"人物1的身份":"..."},{"人物2的身份":"..."}],"具体规则":[{"发言的人":"...","向谁发言"："...","行为prompt"："...","需要的技能":"..."},{"发言的人":"...","向谁发言"："...","行为prompt"："...","需要的技能":"..."}]]' \
                    '你需要翻译的规则为：' + rules
        self.message = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user}
        ]
        while True:
            try:
                rules_ed = self.response_LLMs()
                rules_ed = json.loads(rules_ed)
                break
            except:
                print('Json错误')
        # self.log.write(rules_ed+'\n')
        return rules_ed
