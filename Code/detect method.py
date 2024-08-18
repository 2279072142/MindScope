import json
import random
import ast
import pandas as pd
from LLM.azure_openai import *
from AgentSet.RoleAgent import RoleAgent
from AgentSet.UniversalAgent import checkBiasAgent,BaseAgent
from PromptSet.Prompts_library import *
from LLM.vectorstore import *
import re
from tqdm import tqdm
def solve(text):
    matched = re.search(r"\[(.*?)\]", text)
    extracted_text = matched.group(0) if matched else ""
    return extracted_text
def clean_json_string(s):
    # Remove control characters
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    # Fix common JSON string issues, if necessary, here
    s = re.findall(r'\{.*?\}', s)
    return s[0]



def get_score(agent_score,weights,name):
    score=agent_score[name]['argument support'] * weights[0]
    score+=agent_score[name]['logical consistency']*weights[1]
    score+=agent_score[name]['refutation effectiveness']*weights[2]
    score+=agent_score[name]['argument completeness']*weights[3]
    score+=agent_score[name]['persuasiveness']*weights[4]
    score+=agent_score[name]['reasonability assessment of cognitive bias']*weights[5]
    bias = agent_score[name]['cognitive bias name']
    return score,bias
def debate(left,right,agents,text,judge_agents):

    for i in range(len(agents)):
        agents[i].used_model='gpt-4-turbo'
    allrecord=''
    l_agent=agents[left]
    r_agent=agents[right]
    print(f"{l_agent.bias}与{r_agent.bias} start debate")
    
    l_request=f"you think {l_agent.bias} exists in the current scene. Please introduce to everyone what {l_agent.bias} is. The additional knowledge you can use is {l_agent.knowledge}, limited to 100 words."
    l_response=l_agent.chat(l_request)
    r_agent.receive_message(l_response)
    print(f'{l_agent.name} said:{l_response}\n')
    r_request=f"you think {r_agent.bias} exists in the current scene. Please introduce to everyone what {r_agent.bias} is. The additional knowledge you can use is {r_agent.knowledge}, limited to 100 words."
    r_response = r_agent.chat(r_request)
    l_agent.receive_message(r_response)
    print(f'{r_agent.name} said:{r_response}\n')
    allrecord+=f'{l_agent.name} said:{l_response}\n'
    allrecord += f'{r_agent.name} said:{r_response}\n'

    l_request=f"you think {l_agent.bias} exists in the current scene. Please demonstrate why {l_agent.bias} exists in {text}. The attribute that requires special attention for this cognitive bias is {l_agent.object}. The limit is 100 words. "
    l_response = l_agent.chat(l_request)
    r_agent.receive_message(l_response)
    print(f'{l_agent.name} said:{l_response}\n')
    r_request = f"you think {r_agent.bias} exists in the current scene. Please demonstrate why {r_agent.bias} exists in {text}. The attribute that requires special attention for this cognitive bias is {r_agent.object}, and the limit is 100 words. "
    r_response = r_agent.chat(r_request)
    l_agent.receive_message(r_response)
    print(f'{r_agent.name} said:{r_response}\n')
    allrecord += f'{l_agent.name} said:{l_response}\n'
    allrecord += f'{r_agent.name} said:{r_response}\n'
    
    l_request = f"You think {l_agent.bias} exists in the current scene, and the other party thinks there is {r_agent.bias}, please refute the other party's point of view, limit 200 words"
    l_response = l_agent.chat(l_request)
    r_agent.receive_message(l_response)
    print(f'{l_agent.name} said:{l_response}\n')
    r_request = f"You think {r_agent.bias} exists in the current scene, and the other party thinks there is {l_agent.bias}, please refute the other party's point of view, limit 200 words"
    r_response = r_agent.chat(r_request)
    l_agent.receive_message(r_response)
    print(f'{r_agent.name} said:{r_response}\n')
    allrecord += f'{l_agent.name} said:{l_response}\n'
    allrecord += f'{r_agent.name} said:{r_response}\n'
   
    l_request = f"Based on the debate process between you and your opponent, please summarize the reasons why you think {l_agent.bias} exists in the current scene {text}. Limit 200 words."
    l_response = l_agent.chat(l_request)
    r_agent.receive_message(l_response)
    print(f'{l_agent.name} said:{l_response}\n')
    r_request = f"Based on the debate process between you and your opponent, please summarize the reasons why you think {r_agent.bias} exists in the current scene {text}. Limit 200 words."
    r_response = r_agent.chat(r_request)
    l_agent.receive_message(r_response)
    print(f'{r_agent.name} said:{r_response}\n')
    allrecord += f'{l_agent.name} said:{l_response}\n'
    allrecord += f'{r_agent.name} said:{r_response}\n'

    score_agent1=judge_agents[0]
    score_agent2=judge_agents[1]
    # score,flag=score_agent.scoring(allrecord,l_agent.name,r_agent.name,l_agent.bias,r_agent.bias,text)

    lscore_tmp, lflag = score_agent1.scoring_one(allrecord, l_agent.name, l_agent.bias, text, temperature=0.3)
    rscore, rflag = score_agent2.scoring_one(allrecord, r_agent.name, r_agent.bias, text, temperature=0.5)
    rscore_tmp, rflag = score_agent1.scoring_one(allrecord, r_agent.name, r_agent.bias, text, temperature=0.3)
    lscore, lflag = score_agent2.scoring_one(allrecord, l_agent.name, l_agent.bias, text, temperature=0.5)

    lscore[l_agent.name]['argument support'] = (lscore_tmp[l_agent.name]['argument support'] + lscore[l_agent.name]['argument support']) / 2
    lscore[l_agent.name]['logical consistency'] = (lscore_tmp[l_agent.name]['logical consistency'] +
                                                   lscore[l_agent.name]['logical consistency']) / 2
    lscore[l_agent.name]['refutation effectiveness'] = (lscore_tmp[l_agent.name]['refutation effectiveness'] +
                                                        lscore[l_agent.name]['refutation effectiveness']) / 2
    lscore[l_agent.name]['argument completeness'] = (lscore_tmp[l_agent.name]['argument completeness'] +
                                                     lscore[l_agent.name]['argument completeness']) / 2
    lscore[l_agent.name]['persuasiveness'] = (lscore_tmp[l_agent.name]['persuasiveness'] + lscore[l_agent.name][
        'persuasiveness']) / 2
    lscore[l_agent.name]['reasonability assessment of cognitive bias'] = (lscore_tmp[l_agent.name][
                                                                              'reasonability assessment of cognitive bias'] +
                                                                          lscore[l_agent.name][
                                                                              'reasonability assessment of cognitive bias']) / 2

    rscore[r_agent.name]['argument support'] = (rscore_tmp[r_agent.name]['argument support'] + rscore[r_agent.name][
        'argument support']) / 2
    rscore[r_agent.name]['logical consistency'] = (rscore_tmp[r_agent.name]['logical consistency'] +
                                                   rscore[r_agent.name]['logical consistency']) / 2
    rscore[r_agent.name]['refutation effectiveness'] = (rscore_tmp[r_agent.name]['refutation effectiveness'] +
                                                        rscore[r_agent.name]['refutation effectiveness']) / 2
    rscore[r_agent.name]['argument completeness'] = (rscore_tmp[r_agent.name]['argument completeness'] +
                                                     rscore[r_agent.name]['argument completeness']) / 2
    rscore[r_agent.name]['persuasiveness'] = (rscore_tmp[r_agent.name]['persuasiveness'] + rscore[r_agent.name][
        'persuasiveness']) / 2
    rscore[r_agent.name]['reasonability assessment of cognitive bias'] = (rscore_tmp[r_agent.name][
                                                                              'reasonability assessment of cognitive bias'] +
                                                                          rscore[r_agent.name][
                                                                              'reasonability assessment of cognitive bias']) / 2
    #SAA weight
    #weights=[0.53168381, 0.134733, 0.00000001, 0.04515829, 0.00989386, 0.27853105]


    #RL_Weight
    weights=[1.13262848, 0.24727544,-0.90161614,-0.08157856, 0.53244014, 0.07085065]
    #GA weight
    #weights=[ 0.48802737, 0.13139211,-0.51171129, 0.2586362 , 0.45087349,0.18278212]
    lagent_score,lbias=get_score(lscore,weights,l_agent.name)
    ragent_score,rbias=get_score(rscore,weights,r_agent.name)
    if lagent_score>=ragent_score:
        return left
    else:
        return right


def compete(left,right,agents,text,referee_agents):
    if agents[left].check==True and agents[right].check==True:
        return debate(left, right, agents, text, referee_agents)
    else:
        if agents[left].check==False:
            left_res=agents[left].checkself(text)
            agents[left].check = True
            agents[left].eval=left_res
        else:
            left_res=agents[left].eval
        if agents[right].check==False:
            right_res=agents[right].checkself(text)
            agents[right].check = True
            agents[right].eval=right_res
        else:
            right_res = agents[right].eval

        if left_res['eval']=='yes' and right_res['eval']=='yes':
            return debate(left,right,agents,text,referee_agents)
        elif left_res['eval']=='yes':
            return left
        elif right_res['eval']=='yes':
            return right
        else:
            return -1

def Loser_tree(agents,text,referee_agents):
    num_agents = len(agents)
    tree_size = 2 * num_agents
    tree = [0] * tree_size

    for i in range(num_agents):
        tree[num_agents + i] = i

    for i in range(num_agents - 1, 0, -1):
        left = tree[2 * i]
        right = tree[2 * i + 1]
        if(left==-1 and right==-1):
            tree[i]=-1
        elif left==-1:
            tree[i]=right
        elif right==-1:
            tree[i]=left
        else:
            condition=compete(left, right, agents, text,referee_agents)
            if(condition==right):
                tree[i] = right
            elif(condition==left):
                tree[i]=left
            else:
                tree[i]=-1

    if tree[1]!=-1 and agents[tree[1]].check==False:
        res=agents[tree[1]].checkself(text)
        if res['eval']=='no':
            tree[1]=-1

    if tree[1]==-1:
        return -1
    return agents[tree[1]].bias

def roughcheck_bias(text):

    bias_list = pd.read_excel('./Data/cognitive_bias_v2.xlsx')['biasname']
    bias_text = ''
    for bias in bias_list:
        bias_text += bias + '\n'
    checkAgent = BaseAgent('gpt-4-turbo', './Key_GPT_0.txt')
    checkAgent.system = f'''You are an expert in identifying potential cognitive biases in scene texts. '''

    # checkAgent.user = f'''You will have a list of cognitive biases. Please detect a scene text, Step by step reasoning to determine if there is any cognitive bias present in the list and Return 8 possible bias names in list format, with a decreasing likelihood of occurrence output.The specific content of the list is:<{bias_text}>.\n
    #     Please check the following scenarios:{text},output format: ["xxx","xxx","xxx","xxx","xxx","xxx","xxx","xxx"].Just output the bias name without explaining the reason,Please note that the output cognitive bias names must be from the list of cognitive bias names and ensure that they are exactly consistent with the names in the list'''

    checkAgent.user = f'''You will have a list of cognitive biases. Please detect a scene text, Step by step reasoning to determine if there is any cognitive bias present in the list and Return 5 possible bias names in list format, with a decreasing likelihood of occurrence output.The specific content of the list is:<{bias_text}>.\n
            Please check the following scenarios:{text},output format: ["xxx","xxx","xxx","xxx","xxx"].Just output the bias name without explaining the reason,Please note that the output cognitive bias names must be from the list of cognitive bias names and ensure that they are exactly consistent with the names in the list'''

    checkAgent.setMessage()

    response = checkAgent.response_LLMs(temperature=0)
    check_bias = response.lower()
    check_bias1 = ast.literal_eval(solve(check_bias))

    print(check_bias1)
    # response = checkAgent.response_LLMs(temperature=0.5)
    # check_bias = response.lower()
    # check_bias2 = ast.literal_eval(solve(check_bias))
    # print(check_bias2)
    response = checkAgent.response_LLMs(temperature=0.9)
    check_bias = response.lower()
    check_bias3 = ast.literal_eval(solve(check_bias))
    print(check_bias3)


    check_bias=[]
    isonly=set()
    for i in range(len(check_bias1)):
        isonly.add(check_bias1[i])
        check_bias.append(check_bias1[i])
    for i in range(len(check_bias3)):
        if(check_bias3[i] not in isonly):
            isonly.add(check_bias3[i])
            check_bias.append(check_bias3[i])
    return check_bias
def main():
    data=pd.read_excel('./Data/res_testsetreal_multi-agent-framework.xlsx')
    case=data['case']
    response =data['GPT4']
    label=data['label']
    check_biases=data['res']
    ans=[]
    DB = init_DB()
    with open('./Data/CB_Object.json', 'r') as file:
        CB_Object = json.load(file)
    caserecord=[]
    costrecord=[]
    for i in tqdm(range(0,len(case))):
        text=case[i]
        print(f"Now test {i} case")
        #check_bias=check_biases[i]
        check_bias=roughcheck_bias(text)
       #check_bias = ast.literal_eval(solve(check_bias))

        
        referee_agents=[]
        score_agent1 = checkBiasAgent(model='gpt-4-turbo', personality=judge_role1)
        score_agent2 = checkBiasAgent(model='gpt-4-turbo', personality=judge_role2)
        referee_agents.append(score_agent1)
        referee_agents.append(score_agent2)
       
        elements=[]
        needcheck_bias=[]
        print(check_bias)
        for j in range(0,len(check_bias)):

            for item in CB_Object:
                if(item['bias_name'].lower()==check_bias[j].lower()):
                    needcheck_bias.append(check_bias[j].lower())
                    elements.append(item['elements'])
        print(needcheck_bias)
        Agents = [checkBiasAgent(name=names[j].lower(), bias=needcheck_bias[j], knowledge=get_Knowledge(needcheck_bias[j], DB),object=elements[j],model='gpt-4-turbo') for j in range(len(needcheck_bias))]
        print(f"init {len(Agents)} Agent")
        res=Loser_tree(Agents,text,referee_agents)
        if res==-1:
            ans.append("no bias")
            print('no bias')
        else:
            print(res)
            ans.append(res)
        caserecord.append(text)
        pd.DataFrame({'case':caserecord,'res':ans}).to_excel('./res_detect_method.xlsx')

if __name__ == '__main__':
    main()