from AgentSet.UniversalAgent import *
from AgentSet.RoleAgent import *
import PromptSet.Prompts_library as PL

class RuleStream:
    def __init__(self,origin,target,purpose,content,trans_way):
        self.origin=origin 
        self.target=target  
        self.purpose=purpose 
        self.content=content 
        self.trans_way=trans_way   

    def findAgent(self,name,Roles):
        for role in Roles:
            if role.name==name:
                return role
        return None

    def Broadcast(self,name,Roles,content):
        for role in Roles:
            if role.name!=name:
                role.receive_message(name+" said："+content)

    def run(self,Roles,log):
        orgin_agent=self.findAgent(self.origin,Roles)
        if self.trans_way=='self-receiving':  
            target_agent=self.findAgent(self.target,Roles)
            target_agent.receive_message(self.content)
            PL.records_dynatic.append({'origin':'system','target':target_agent.name,'trans_way':'self-receiving','content':self.content})
            log.write("#############################\n")
            log.write(target_agent.name+' receive:'+self.content+'\n')
            log.write("#############################\n")
        elif self.trans_way=='broadcast': 
            response = orgin_agent.chat(self.content)
            PL.records_dynatic.append(
                {'origin': orgin_agent.name, 'target': 'all', 'trans_way': 'broadcast',
                 'content': response})
            log.write("#############################\n")
            log.write(orgin_agent.name+' broadcasting to everyone,saying:'+response+'\n')
            log.write("#############################\n")
            self.Broadcast(self.origin,Roles,response)
        else:   
            if self.trans_way=='unicast':
                #self.content+=PL.nine_basic_communicative_purposes[self.purpose]
                if self.target=='system':
                    response = orgin_agent.chat(self.content)
                    PL.records_dynatic.append(
                        {'origin': orgin_agent.name, 'target': 'system', 'trans_way': 'unicast',
                         'content': response})
                    log.write("#############################\n")
                    log.write(orgin_agent.name+' to system,saying:'+response+"\n")
                    log.write("#############################\n")
                else:
                    if isinstance(self.target,list):
                        self.target=self.target[0]
                        target_agent=self.findAgent(self.target,Roles)
                    else:
                        target_agent = self.findAgent(self.target, Roles)
                    response=orgin_agent.chat(self.content)
                    PL.records_dynatic.append(
                        {'origin': orgin_agent.name, 'target': target_agent.name, 'trans_way': 'unicast',
                         'content': response})
                    log.write("#############################\n")
                    log.write(orgin_agent.name+' to '+target_agent.name+',saying:'+response+'\n')
                    log.write("#############################\n")
                    target_agent.receive_message(self.origin+" said："+response)
            else: #Multicast
                #self.content += PL.nine_basic_communicative_purposes[self.purpose]
                response = orgin_agent.chat(self.content)
                targets_name=''
                for tar_role in self.target:
                    target_agent = self.findAgent(tar_role,Roles)
                    if target_agent !=None:
                        targets_name += target_agent.name + ','
                        target_agent.receive_message(self.origin + " said：" + response)
                PL.records_dynatic.append(
                    {'origin': orgin_agent.name, 'target': targets_name[:-1], 'trans_way': 'multicast',
                     'content': response})
                log.write("#############################\n")
                log.write(orgin_agent.name + ' to ' + targets_name[:-1] + ',saying:' + response + '\n')
                log.write("#############################\n")
