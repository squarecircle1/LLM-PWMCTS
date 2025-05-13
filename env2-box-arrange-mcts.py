from LLM import *
from prompt_env2 import *
from env2_create import *
from MCTS_env2 import *
from sre_constants import error
from database_build import DocumentSearch
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import pprint
import sys

def write_in_log(feedback_list, response, state_update_prompt, index_query_times, token_num_count_list_add, log_path):
  if len(feedback_list)>0:
    with open(log_path + '/Process_Recording.txt', 'a') as wf:
      wf.write(f'\n***** Step #{index_query_times+1} *****\n')
      wf.write(f'\n#####\n\nState #{index_query_times+1}:\n{state_update_prompt}\n')
      for d in feedback_list:
        wf.write(f'\naction :\n{d["action"]}\nfeedback :\n{d["feedback"]}\nreason :\n{d["note"]}\n')
    with open(log_path + '/feedback_token.txt', 'a') as wf:
      for d in feedback_list:
        wf.write(str(d["token"]) + '\n')
    with open(log_path + '/replan_token.txt', 'a') as wf:
      for d in token_num_count_list_add:
        wf.write(str(d) + '\n')
    with open(log_path + '/feedback_'+str(index_query_times+1)+'.json', 'w') as f:
      json.dump(feedback_list, f)
  else:
    with open(log_path + '/Process_Recording.txt', 'a') as wf:
      wf.write(f'\n***** Step #{index_query_times+1} *****\n')
      wf.write(f'\n#####\n\nState #{index_query_times+1}:\n{state_update_prompt}\n')
      wf.write(f'\naction :\n{response}\n')


def run_exp_score(pg_state_list, step_num):
    pg_dict_initial = pg_state_list[0]
    remaining_box_dict = pg_state_list[-1]
    boxes_all_list = [item for items in pg_dict_initial.values() for item in items if item.startswith('box')]
    boxes_remaining_list = [item for items in remaining_box_dict.values() for item in items if item.startswith('box')]
    lifted_weight_ratio = (len(boxes_all_list) - len(boxes_remaining_list)) / len(boxes_all_list)
    # lifted_weight_ratio = lifted_weight_ratio / (0.1*step_num)
    return lifted_weight_ratio


def response_dict_filter_agent(system_error_feedback, original_response_dict):
  segments = system_error_feedback.split(';')
  # 提取数值
  values = []
  for segment in segments:
    # 去掉前后空格
    segment = segment.strip()
    # 查找数值
    match = re.search(r'(\d+\.\d+)_(\d+\.\d+)', segment)
    if match:
        values.append((float(match.group(1)), float(match.group(2))))

  for value in values:
    agent_name = f'Agent[{value[0]}, {value[1]}]'
    del original_response_dict[agent_name]

  final_actions = {}
  for agent, action in original_response_dict.items():
    box = action.split(",")[0].split("(")[1]
    final_actions[box] = (agent, action)
  filtered_response_dict = {agent: action for agent, (agent, action) in final_actions.values()}
  return filtered_response_dict


def run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, simulation_num, cen_decen_framework = 'test', model_name='deepseek-chat'):
    Saving_path_result = Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}_{model_name}'

    # specify the path to your dir for saving the results
    os.makedirs(Saving_path_result, exist_ok=True)
    os.makedirs(Saving_path_result+f'/response', exist_ok=True)
    os.makedirs(Saving_path_result+f'/pg_state', exist_ok=True)

    with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
        pg_dict = json.load(file)

    response_total_list = [] # The record list of all the responses
    pg_state_list = [] # The record list of apg states in varied steps

    pg_state_list.append(pg_dict)

    with open(Saving_path_result+'/pg_state' + '/pg_state'+str(1)+'.json', 'w') as f:
        json.dump(pg_dict, f)

    search_engine = DocumentSearch(model_name="sentence-transformers/all-MiniLM-L6-v2")
    env = MCTS_env(pg_dict, pg_row_num, pg_column_num, model_name)
    agent = MCTSAgent(env=env, max_depth=query_time_limit, simulation_num=simulation_num, uct_type='PUCT', search_engine=search_engine,  use_llm = False, use_b=True)

    history = []
    done = False
    ### Start the Game! Query LLM for response
    print(f'query_time_limit: {query_time_limit}')
    for index_query_times in range(query_time_limit): # The upper limit of calling LLMs
        #print(index_query_times)
        print(f'-------state_{index_query_times+1}:---------')
        pprint.pprint(pg_dict)
        agent.belief_dict = agent.gen_belief_dict(pg_dict)

        action = agent.search(pg_dict, history, done, Saving_path_result, index_query_times)
        print("action:\n")
        pprint.pprint(action)

        with open(Saving_path_result+'/response' + '/response'+str(index_query_times+1)+'.json', 'w') as f:
          json.dump(action, f)

        try:
          system_error_feedback, pg_dict_returned, collision_check = action_from_response(pg_dict, action)
           # 如果system_error_feedback不为空，说明出现了一个box被多个agent移动了
          if system_error_feedback != '':
            print('----- action_from_response -----')
            print(system_error_feedback)
          if collision_check:
            print('Collision!')
            success_failure = 'Collision'
            return response_total_list, pg_state_list, success_failure, index_query_times, Saving_path_result
          
          response_total_list.append(action)
          pg_dict = pg_dict_returned
          # valid_actions = agent.env.get_valid_action(pg_dict)
        except:
          success_failure = 'Hallucination of wrong plan'
          return response_total_list, pg_state_list, success_failure, index_query_times, Saving_path_result

        pg_state_list.append(pg_dict)
        with open(Saving_path_result+'/pg_state' + '/pg_state'+str(index_query_times+2)+'.json', 'w') as f:
            json.dump(pg_dict, f)

        # Check whether the task has been completed
        count = 0
        for key, value in pg_dict.items():
          count += len(value)
        if count == 0:
          break

    if index_query_times < query_time_limit - 1:
      success_failure = 'success'
    else:
      success_failure = 'failure over query time limit'
    return response_total_list, pg_state_list, success_failure, index_query_times, Saving_path_result

Code_dir_path = './data_test/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env2_BoxNet2'
model_name = 'gpt-4o-2024-08-06'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613'
print(f'-------------------Model name: {model_name}-------------------')

for pg_row_num, pg_column_num in [(4,4)]:
  if pg_row_num == 4 and pg_column_num == 8:
    query_time_limit = 20
  else:
    query_time_limit = 20
  simulation_num = 50

  framework='llm_mcts_pw_1'
  for iteration_num in range(2,3):
    print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

    response_total_list, pg_state_list, success_failure, index_query_times, Saving_path_result = run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, simulation_num, cen_decen_framework=framework, model_name = model_name)
    print(success_failure)
    print(f'Iteration number: {index_query_times+1}')

    with open(Saving_path_result + '/success_failure.txt', 'w') as f:
      f.write(success_failure)

    with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
      f.write(f'{index_query_times+1}')
