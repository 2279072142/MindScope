from AgentSet.UniversalAgent import *
import pandas as pd
from tqdm import tqdm
def main():
    data=pd.read_excel('../Data/res_testsetreal_multi-16-prompt.xlsx')
    label=data['label']
    case=data['case']
    azure = Azure('xx',
                  'xx',
                  './Token/access_token.txt', './Token/refresh_token.txt')
    ans=[]
    for i in tqdm(range(0,len(case))):
        print(f"Now test {i} case")
        text=case[i]
        bias_list = pd.read_excel('../Data/cognitive_bias_v2.xlsx')['biasname']
        bias_text = ''
        for bias in bias_list:
            bias_text += bias + '\n'
        checkAgent = checkBiasAgent(azure=azure)
        checkAgent.system = f'''You are an expert in identifying potential cognitive biases in scene texts. '''

        checkAgent.user = f'''You will have a list of cognitive biases. Please detect a scene text, Step by step reasoning to determine if there is any cognitive bias present in the list and Return 16 possible bias names in list format, with a decreasing likelihood of occurrence output.The specific content of the list is:<{bias_text}>.\n
               Please check the following scenarios:{text},output format: ["xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx"].Just output the bias name without explaining the reason,Please note that the output cognitive bias names must be from the list of cognitive bias names and ensure that they are exactly consistent with the names in the list'''
        checkAgent.setMessage()
        request=[
            {"role": "system", "content": checkAgent.system},
            {"role": "user", "content":checkAgent.user}
        ]
        candidateset = checkAgent.response_LLMs(request)
        print(candidateset)
        checkAgent.user=f'''After your initial screening, you have found that there may be candidate sets of cognitive biases in the scene. Please carefully plan and reflect on how to determine whether there is cognitive bias in the scene.Limit 100 words\n
                        The candidate set for cognitive bias is {candidateset} \nScene is {text}'''
        request = [
            {"role": "system", "content": checkAgent.system},
            {"role": "user", "content": checkAgent.user}
        ]
        reflectandplan=checkAgent.response_LLMs(request)
        print(reflectandplan)
        checkAgent.user=f'''Based on your plan and reflection, carefully consider whether there is one of these cognitive bias candidate sets in this scenario. If so, output the name of the most likely cognitive bias. If not, output "no bias"\n
                        Your plan and reflection are: {reflectandplan} \nScenario is: {text} \nCognitive bias candidate set: {candidateset}
                        '''
        request = [
            {"role": "system", "content": checkAgent.system},
            {"role": "user", "content": checkAgent.user}
        ]
        res=checkAgent.response_LLMs(request)
        print(res)
        ans.append(res)

if __name__ == '__main__':
    main()