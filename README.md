# README

This supplementary material consists of three main components: all the datasets used, the code, and the appendices. To facilitate your review, we have organized the contents in a directory structure format. Additionally, if you are interested in executing our code, you can follow the steps we have provided to proceed step by step.

## Dataset

- MindScope

  - static_dataset.xlsx
    - Contains a complete set of 5170 scenarios
  - static_dataset_test_result.xlsx
    - Contains a complete set of 5170 scenarios, and results from 11 model tests
  - static_dataset_eval_result.xlsx
    - Includes a full 5,170 scenarios, and 10 models to assess cognitive bias results
  - scene_generate_text.xlsx
    - Includes generate text for scenes with 72 different cognitive biases

  - dynamic_dataset.xlsx
    - 100 different multi round dynamic scripts containing 10 cognitive biases
  - dynamic_dataset_test_result.xlsx
    - Dynamic Dataset Test Results
  - dynamic_dataset_eval_result.xlsx
    - Dynamic dataset evaluation results

- detect_element.json

  - Elements of cognitive bias detection

- classic_case_library.xlsx

  - Classic Case Knowledge Base

- debate_record.xlsx

  - Decision module training data

- testset4_without_label.xlsx

  - Unlabeled detection dataset

  **Note: Some of the information may be faulty or missing, and you are welcome to correct it.**


## Appendix

- **Dataset Construction**
  - **A. Static dataset construction**
  - **B. Dynamic dataset construction**
- **Experimental setup**
  - **C. Proficiency testing of GPT-4 as an evaluator**
  - **D. Rule-Based Multi-Agent Communication**
  - **E. Cognitive bias in different LLMs**
    - **E.1 Cognitive bias detection in static dataset**
    - **E.2 Cognitive bias detection in dynamic dataset**
  - **F. Method for Detecting Cognitive Bias Without Labels**
    - **F.1 Cognitive bias detection of existing methods**
    - **F.2 Ablation experiments**
    - **F.3 Decision module training**

## Code

### Installation

- Creating a virtual environment for Python

  ```
  conda create -n MindScope python=3.9 -y
  ```

- Install the necessary packages

  ```
  pip install -r requirements.txt
  ```

- Setting the Openai key

  - Place your sk-xxx openai api in the root directory under **Key_GPT_0.txt**.

  

- Example

  - test Static dataset

    - gpt-3.5-turbo
  
      ```
      python test_LLM.py -model_name 'GPT' -model_type 'gpt-3.5-turbo'
      ```
  
    - gpt-4-turbo
  
      ```
      python test_LLM.py -model_name 'GPT' -model_type 'gpt-4-turbo'
      ```
  
    - llama2-7b   local 
  
      ```
      python test_LLM.py -model_path 'your model weight path' -model_name 'Llama2' 
      ```
  
    - llama3-8b  local 
  
      ```
      python test_LLM.py -model_path 'your model weight path' -model_name 'Llama3' 
      ```
  
    - chatglm-6b  local 
  
      ```
      python test_LLM.py -model_path 'your model weight path' -model_name 'ChatGLM' 
      ```
  
    - vicuna-7b  local 
  
      ```
      python test_LLM.py -model_path 'your model weight path' -model_name 'Vicuna'
      ```
  
  - Evaluate cognitive biases in LLMs (In static dataset)
  
    - llama3-8b
  
      ```
      python evaluate_LLM.py --test_model 'llama3-8B' --used_model 'gpt-4-turbo'
      ```
  
    - llama3-70b
  
      ```
      python evaluate_LLM.py --test_model 'llama3-70B' --used_model 'gpt-4-turbo'
      ```
  
    - llama2-7b
  
      ```
      python evaluate_LLM.py --test_model 'llama2-7B' --used_model 'gpt-4-turbo'
      ```
  
    - llama2-13b
  
      ```
      python evaluate_LLM.py --test_model 'llama2-13B' --used_model 'gpt-4-turbo'
      ```
  
    - llama2-70b
  
      ```
      python evaluate_LLM.py --test_model 'llama2-70B' --used_model 'gpt-4-turbo'
      ```
  
    - gpt-3.5-turbo
  
      ```
      python evaluate_LLM.py --test_model 'gpt-3.5' --used_model 'gpt-4-turbo'
      ```
  
    - gpt-4-tubo
  
      ```
      python evaluate_LLM.py --test_model 'GPT4' --used_model 'gpt-4-turbo'
      ```
  
    - chatglm
  
      ```
      python evaluate_LLM.py --test_model 'chatglm-6b' --used_model 'gpt-4-turbo'
      ```
  
    - vicuna-7b
  
      ```
      python evaluate_LLM.py --test_model 'vicuna-7b' --used_model 'gpt-4-turbo'
      ```
  
    - vicuna-13b
  
      ```
      python evaluate_LLM.py --test_model 'vicuna-13b' --used_model 'gpt-4-turbo'
      ```
  
    - vicuna-33b
  
      ```
      python evaluate_LLM.py --test_model 'vicuna-33b' --used_model 'gpt-4-turbo'
      ```
  
  - test Dynamic dataset
  
    ```
    #default test gpt-4-turbo
    python RuleGen.py
    ```
  
  - Evaluate cognitive biases in LLMs (In dynamic dataset)
  
    ```
    python evaluateCB_LLM_Dynamic.py
    ```
  
  - build debate set
  
    ```
    python build_debate_set.py
    ```
  
  - Method for Detecting Cognitive Bias (Without Labels)
  
    ```
    python detect_method.py
    ```
  
  - Training Decision Module
  
    ```
    python Train_decision_module/ant_colony_optimization.py
    python Train_decision_module/genetic_algorithm.py
    python Train_decision_module/RL_DQN_log.py
    python Train_decision_module/simulated_annealing.py
    ```

**If you need other codes, feel free to contact us in the future!**

