nine_basic_communicative_purposes={
    "情境相关评论":"当说话者在谈话中评论在场的人或物，或在他们共同的情景上下文中发生的事件时，就会出现这种情况。这方面的例子包括(1)在加油站等待加油的另一名司机的不安全驾驶行为的评论，以及(2)在玩棋盘游戏时谈论规则和策略。",
    "开玩笑":"这包括以幽默为目的的对话，包括轻松幽默和黑色幽默。它还包括幽默的玩笑，挑逗和调情。这方面的例子包括:(1)将难吃的馅饼和锯末进行夸张的比较，(2)一位发言者取笑另一位发言者，因为她的衬衫穿反了。",
    "卷入冲突":"这个目的包括任何形式的分歧，包括更轻松的辩论和更严肃的争吵。例如:(1)关于钥匙圈上的哪把钥匙适合房子的哪扇门的争论;(2)关于两个人中哪一个更有可能有一天变得富有的友好分歧。",
    "辨识":"这个目的概括了旨在探索或考虑选项或计划的讨论，包括关于事物如何工作以及问题的最佳解决方案可能是什么的讨论。例子包括(1)讨论在配偶去世后拜访亲家是否合适，以及(2)试图理解和解释共同认识的人最近的行为。",
    "分享感受和评价":"这包括关于感受、评价、意见和信仰的讨论，包括表达不满和分享个人观点。例如:(1)解释讲话者对某件衣服的偏好，以及(2)关于政治观点的讨论。",
    "提供建议和指示":"当一个说话者向另一个说话者提供方向、建议或建议时，就会出现这种情况。例子包括:(1)一个演讲者通过在过程中给出一步一步的指令来帮助另一个人浏览网站来订购门票，(2)一个演讲者就购买最好的复印纸提供建议。",
    "描述或解释过去的":"这一目的包括关于过去真实事件的叙事性故事或其他关于过去的人或事件的参考。例如:(1)一位讲话者讲述他最喜欢的假期的故事;(2)两位讲话者一边整理阁楼上的箱子一边回忆过去。",
    "描述或解释未来":"这包括对未来事件和意图的描述或推测，包括那些计划好的和那些更具假设性的。例子包括:(1)一位演讲者描述了与重要的另一半约会的计划，以及(2)两位演讲者分享了他们大学毕业后的生活计划。",
    "描述或解释(时间中立)":"对时间(过去或未来)不相关或不明确的事实、信息、人物或事件的描述或解释。例如:(1)一个说话者回答另一个人关于房屋装修进度的问题，(2)描述两种产品之间的区别。"
}

initrole_prompt='''
Please pay attention to following the requirements below
1. Please use your character's perspective in the conversation and do not participate as artificial intelligence in the conversation
2. The word limit is 100 words
3. Do not answer questions in a tone other than your own character
4. Just answer the content directly, do not start with<your own name:>when answering the content
'''

get_role_prompt='''
Now you need to extract the information in the following scenario and output the results in the specified json format. For example:
The following is a specific scenario:
## education field
### Scene background
A new educational institution called Bright Future plans to launch a new online education platform that aims to improve students' academic performance through personalized learning plans. The agency's two key decision-makers are deciding whether to proceed with the project based on available market data and personal judgment.
### Character setting
1. The founder and education project manager (Jordan) is passionate about online education and personalized learning, believing that this is the future of education reform.
2. The sponsor and board member of the Education Foundation (Casey) is concerned about the return and sustainability of educational investment and has reservations about the effectiveness of online education.
### Scenario rules
1. Jordan and Casey will receive various information about the market and the effectiveness of online education, which may be positive or negative.
2. Jordan needs to explain to Casey why the new platform has the potential to improve the quality of education and seek Casey’s support.
3. Casey needs to decide whether to support this project based on Jordan's proposal and his own judgment.
4. Both parties need to clearly communicate the basis for their decision-making.
5. After each round of communication, characters can adjust their stance based on new information.
### Tasks for each character
Jordan: Need to provide persuasive market and product analysis to prove the educational value of the new platform and seek support from Casey.
Casey: Jordan’s proposal needs to be evaluated, market demand and potential educational effects need to be considered to make investment decisions.
### Scene information
Information 1: Market data\nMarket research shows that due to the COVID-19 pandemic, many families and schools have gradually accepted online education, and the online education market is expected to continue to grow in the next few years. However, there are many mature online education platforms on the market, and competition is fierce. \nMessage 2: Personalized learning effects\nSome studies have shown that personalized learning plans can help students find a learning pace that suits them, thereby improving academic performance. However, some studies have pointed out that the lack of face-to-face communication and guidance may affect students' learning effects and social skills. Information 3: Return on Investment\nPreliminary budget analysis shows that in order to develop and operate this new platform, Bright Future will require a significant initial investment. While it may be profitable in the long term, you may not see a clear return on investment in the short term. \nMessage 4: Success and Failure Cases\nIn recent years, some online education platforms such as Khan Academy and Coursera have achieved significant success, while some small or start-up online education companies have failed due to lack of funds and experience. \nMessage 5: Comparison of traditional education and online education\nSome education experts believe that although online education provides a flexible and convenient way of learning, traditional face-to-face education is irreplaceable in cultivating students' social skills and teamwork abilities. Advantage. \nMessage 6: Technological Progress\nWith the advancement of AI and big data technology, new online education platforms have the potential to provide more accurate personalized learning plans and real-time feedback, thereby improving the quality of education and student experience. \nInformation 7: Social Opinion\nSocial opinion shows that some parents and educators have reservations about online education, fearing that it may exacerbate the problems of educational injustice and uneven distribution of resources. \n".

Extraction format:
The scene contains the role of the scene, the background of the role, and the role of the role. Please extract and process it in the following format.
[
     {
         "name":"Jordan",
         "background": "You are the founder and education project manager of an educational institution. You are passionate about online education and personalized learning, and believe that this is the future of education reform.",
         "task": "Need to provide persuasive market and product analysis to demonstrate the educational value of the new platform and seek Casey's support."
     },
     {
         "name":"Casey",
         "background": "You are a funder and board member of an education foundation. You are concerned about the return and sustainability of education investment, and have reservations about the effectiveness of online education.",
         "task": "Need to evaluate Jordan's proposal, consider market demand and potential educational effects, and make investment decisions."
     }
]
The key value for generating JSON must be lowercase
Now I want to give you a new scenario. Please output the content in json format according to the above output format. Note that you only need to output the content in json format.
The new scene is:
'''

# get_rule_prompt='''
# Now you need to extract the information in the following scenario and output the results in the specified json format. For example:
# The following is a specific scenario:
# ## education field
# ### Scene background
# A new educational institution called Bright Future plans to launch a new online education platform that aims to improve students' academic performance through personalized learning plans. The agency's two key decision-makers are deciding whether to proceed with the project based on available market data and personal judgment.
# ### Character setting
# 1. The founder and education project manager (Jordan) is passionate about online education and personalized learning, believing that this is the future of education reform.
# 2. The sponsor and board member of the Education Foundation (Casey) is concerned about the return and sustainability of educational investment and has reservations about the effectiveness of online education.
# ### Scenario rules
# 1. Jordan and Casey will receive various information about the market and the effectiveness of online education, which may be positive or negative.
# 2. Jordan needs to explain to Casey why the new platform has the potential to improve the quality of education and seek Casey’s support.
# 3. Casey needs to decide whether to support this project based on Jordan's proposal and his own judgment.
# 4. Both parties need to clearly communicate the basis for their decision-making.
# 5. After each round of communication, characters can adjust their stance based on new information.
# ### Tasks for each character
# Jordan: Need to provide persuasive market and product analysis to prove the educational value of the new platform and seek support from Casey.
# Casey: Jordan’s proposal needs to be evaluated, market demand and potential educational effects need to be considered to make investment decisions.
# ### Scene information
# Information 1: Market data\nMarket research shows that due to the COVID-19 pandemic, many families and schools have gradually accepted online education, and the online education market is expected to continue to grow in the next few years. However, there are many mature online education platforms on the market, and competition is fierce. .
#
# Extraction format:
# The scene includes the rules of the scene, including multiple interactive contents, including the initiating object, the receiving object, the purpose of interaction (including answering, asking, receiving information, summarizing opinions), information content, and communication methods (including unicast, broadcast, group broadcast, self-receive). Please follow the order of scene rules and output in the following format:
# [
#      {
#          "initiating":"Jordan",
#          "receive":"Jordan",
#          "purpose": "Receive information from the system",
#          "content":"Information 1: Market data\nMarket research shows that due to the COVID-19 pandemic, many families and schools have gradually accepted online education, and the online education market is expected to continue to grow in the next few years. However, the market has There are many mature online education platforms and competition is fierce.",
#          "propagation":"self-receiving"
#      },
#      {
#          "initiating":"Casey",
#          "receive":"Casey",
#          "purpose": "Receive information from the system",
#          "content":"Information 1: Market data\nMarket research shows that due to the COVID-19 pandemic, many families and schools have gradually accepted online education, and the online education market is expected to continue to grow in the next few years. However, the market has There are many mature online education platforms and competition is fierce.",
#          "propagation":"self-receiving"
#      },
#      {
#          "initiating":"Jordan",
#          "receive":"Casey",
#          "purpose": "Describe or explain the future",
#          "content": "Explain why the new platform has the potential to improve the quality of education and seek Casey's support.",
#          "propagation":"unicast"
#      },
#      {
#          "initiating":"Casey",
#          "receive":"Jordan",
#          "purpose":"identification",
#          "content: "You need to decide whether to support this project based on Jordan's proposal and your own judgment.",
#          "propagation":"Unicast"
#      },
#      {
#          "initiating":"Jordan",
#          "receive":"All",
#          "purpose": "Share feelings and evaluations",
#          "content": "Clearly express the basis for your decision-making",
#          "propagation":"Broadcast"
#      },
#      {
#          "initiating":"Casey",
#          "receive":["Jordan"],
#          "purpose": "Share feelings and evaluations",
#          "content": "Clearly express the basis for your decision-making",
#          "propagation":"Multicast"
#      }
# ]
# The key value for generating JSON must be lowercase
# Now I want to give you a new scenario. Please output the content in json format according to the above output format.
# The new scene is:
# '''
get_rule_prompt='''
# Online Psychological Simulation Protocol
## Scene Purpose
To investigate the presence of the sunk cost fallacy in participants, examining how individuals continue a behavior or endeavor as a result of previously invested resources (time, money, effort).
## Scene Background
The simulation is set in a virtual investment scenario. Participants are led to believe they are part of an online investment group, discussing and making decisions about ongoing investment projects.
## Settings of Roles
### Role One:
- **Name:** Alex
- **Identity:** Subject
- **Background:** Alex is a new member of the online investment group, interested in learning about investing.
- **Task:** Alex's task is to participate in investment discussions and make decisions about whether to continue or abandon different investment projects.
### Role Two:
- **Name:** Jordan
- **Identity:** Confederate
- **Background:** Jordan is an experienced member of the investment group, providing guidance and opinions on investments.
- **Task:** Jordan's task is to subtly encourage Alex to continue investing in a losing project, embodying the sunk cost fallacy.
### Role Three:
- **Name:** Taylor
- **Identity:** Moderator
- **Background:** Taylor oversees the simulation, ensuring smooth communication and adherence to the protocol.
- **Task:** Taylor provides information about the investment scenarios, moderates discussions, and collects data on participants' decisions.
## Scenario Rules
1.	(Self-receiving, receiving information from the system) Subject receives initial information about the investment game and its rules from the system.
2.	(Broadcast, Provide Advice and Instructions) Moderators shares an overview of the virtual investment market's current status with both participants.
3.	(Unicast – Figuring-Things-Out): Subject and Moderator engage in a discussion about their initial investment strategies. 
4.	(Unicast, Figure Things Out) Moderators asks for subjects whether to continue or abandon the investment
5.	(Self-receiving, Receive Information from the System) Subject receives updates from the system about the status and results of the investment projects, indicating a minor loss
6.	(Broadcast, Describing or explaining the future) Confederate shares enthusiasm about the investment projects with Subject and Moderator, highlighting its potential despite initial losses.
7.	(Unicast, Figure Things Out) Moderators asks for subjects whether to continue or abandon the investment
8.	(Self-receiving, Receive Information from the System) Subject receives updates from the system about the status and results of the investment projects, indicating a moderate loss
9.	(Unicast – Giving advice and instructions): Confederate advises Subject on subsequent investment decisions, emphasizing the importance of sticking with initial choices despite negative outcomes. 
10.	(Unicast, providing response) Subject sends a final decision to Moderator, concluding the simulation.
## Method to Observe the Corresponding Cognitive Bias Phenomenon
The primary method of observing the sunk cost fallacy is through Subject's investment decisions. If Subject, despite negative information and poor project performance, continues to invest in the project encouraged by Confederate, it indicates the presence of the sunk cost fallacy. These decisions can be quantitatively assessed by tracking the frequency and conditions under which Subject chooses to continue or abandon investments. This binary observation (continue vs. abandon) provides a clear metric for the presence of the sunk cost fallacy in the participant's decision-making process.

Extraction format:
The scene includes the rules of the scene, including multiple interactive contents, including the initiating object, the receiving object, the purpose of interaction (including answering, asking, receiving information, summarizing opinions), information content, and communication methods (including unicast, broadcast, group broadcast, self-receive). Please follow the order of scene rules and output in the following format:
[
     {
         "initiating":"system",
         "receive":"Alex",
         "purpose": "Receive Information from the System",
         "content":"Subject receives initial information about the investment game and its rules from the system.",
         "propagation":"self-receiving"
     },
     {
         "initiating":"Taylor",
         "receive":"all",
         "purpose": "Provide Advice and Instructions",
         "content": "shares an overview of the virtual investment market's current status with both participants",
         "propagation":"broadcast"
     },
     {
         "initiating":"Alex",
         "receive":"Taylor",
         "purpose":"Figuring Things Out",
         "content: "engage in a discussion about their initial investment strategies.",
         "propagation":"unicast"
     },
     {
         "initiating":"Taylor",
         "receive":"Alex",
         "purpose": "Figure Things Out",
         "content": "asks for subjects whether to continue or abandon the investment",
         "propagation":"unicast"
     },
     {
         "initiating":"system",
         "receive":"Alex",
         "purpose": "Receive Information from the System",
         "content": "receives updates from the system about the status and results of the investment projects, indicating a minor loss",
         "propagation":"self-receiving"
     },
     {
         "initiating":"Jordan",
         "receive":"all",
         "purpose": "Describing or explaining the future",
         "content": "shares enthusiasm about the investment projects with Alex and Jordan, highlighting its potential despite initial losses.",
         "propagation":"broadcast"
     },
     {
         "initiating":"Taylor",
         "receive":"Alex",
         "purpose": "Figure Things Out",
         "content": "asks for subjects whether to continue or abandon the investment",
         "propagation":"unicast"
     },
     {
         "initiating":"system",
         "receive":"Alex",
         "purpose": "Receive Information from the System",
         "content": "receives updates from the system about the status and results of the investment projects, indicating a moderate loss",
         "propagation":"self-receiving"
     },
     {
         "initiating":"Jordan",
         "receive":"Alex",
         "purpose": "Giving advice and instructions",
         "content": "advises Alex on subsequent investment decisions, emphasizing the importance of sticking with initial choices despite negative outcomes. ",
         "propagation":"unicast"
     },
     {
         "initiating":"Alex",
         "receive":["Jordan","Taylor"],
         "purpose": "providing response",
         "content": "sends a final decision to Jordan and Taylor, concluding the simulation.",
         "propagation":"Multicast"
     }
]
The key value for generating JSON must be lowercase and In rule extraction, please convert Subject, Confederate, and Moderator to the names of the corresponding roles, rather than directly using Subject, Confederate, and Moderator
Now I want to give you a new scenario. Please output the content in json format according to the above output format.
The new scene is:
'''
names = [
    'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia',
    'Liam', 'Noah', 'William', 'James', 'Oliver',
    'Charlotte', 'Mia', 'Amelia', 'Harper', 'Evelyn',
    'Benjamin', 'Lucas', 'Mason', 'Ethan', 'Alexander',
    'Abigail', 'Emily', 'Elizabeth', 'Mila', 'Ella',
    'Daniel', 'Henry', 'Jackson', 'Sebastian', 'Aiden'
]

records_dynatic=[]


judge_role1='As an Analytical Judge Agent, I place a great emphasis on facts and data. In judging debates, I rely on detailed data analysis and logical reasoning. I meticulously examine the evidence provided by each debater, assessing its relevance and accuracy, and ensuring that their arguments are logically consistent and tightly-knit. My decision-making process is strictly based on quantitative data and logical frameworks, with high standards for the validity of arguments and the reliability of evidence. I value the consistency of arguments, the accuracy of data, and the completeness of the reasoning.'
judge_role2='As an Intuitive Judge Agent, my judgment is more based on intuition and the overall impression of the debate. In evaluating debates, I focus not only on facts and logic but also on the overall performance and style of the debaters. I am highly sensitive to the dynamics of the debate, the emotional expression of the participants, and their persuasiveness. During the judging process, I rely on my intuition and subjective feelings to assess the quality of the debate, which includes the debaters‘ performance, emotional conveyance, and overall persuasiveness.'
role_prompt='Note: 1. Please always remember your name and personal information background;\n 2. Your answer should be based on your personality characteristics as much as possible, and answer like an ordinary human being;\n 3. Please keep your answer as concise as possible, On the basis of completing the task, the word limit is 100 words;\n 4. Please do not add your name before answering, as this will affect our subsequent operations.\n5.Please act strictly according to your decision-making style.\n6.Do not directly state your decision-making style in your speech'
decision_style_radical='''radical decision-making style
Risk attitude: Radical decision-makers tend to accept high risks and seek high returns.
Change oriented: They are more inclined to drive change, pursue innovation and breakthrough solutions.
Rapid decision-making: In the decision-making process, radical decision-makers often make choices faster and tend to make intuitive judgments.
Competitive advantage: They strive to quickly gain market advantage through decision-making, even if it means taking on greater uncertainty.
Resource allocation: When allocating resources, one is more willing to invest in high-risk but potentially high return projects.
'''
decision_style_conservative='''conservative decision-making style
Risk avoidance: Conservative decision-makers typically avoid high risks and seek stable and reliable returns.
Stability pursuit: They tend to maintain the status quo and prioritize traditional methods that have been proven effective.
Prudent decision-making: Before making a decision, they usually conduct in-depth analysis and evaluation, and the decision-making process is more cautious and slow.
Risk management: Emphasize risk management and mitigation, and strive to ensure that the risks of each decision are within a controllable range.
Resource conservatism: In resource allocation, there is a greater tendency towards low-risk investments and a focus on long-term stable growth.
'''