from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000

def LLM_summarize_func(state_action_prompt_next_initial):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name='gpt-4')
  return response

collision_avoidance_prompt = '[Do remember that each corner can only contain at most one box! Each box can only be moved by one agent! Hence, you need to avoid the collision of boxes. Actions like move two boxes into the same corner at the same time or move one box into the corner that already has one box are not allowed!]'

def input_prompt_1_func(state_update_prompt):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
  '''
  return user_prompt_1


def input_prompt_1_only_state_action_func(state_update_prompt, response_total_list, pg_state_list):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  state_action_prompt = ''
  for i in range(len(response_total_list) - 1, -1, -1):
    state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
    if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
      state_action_prompt = state_action_prompt_next
    else:
      break

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.
  
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  return user_prompt_1


def input_prompt_1_func_total(state_update_prompt, response_total_list,
                                                pg_state_list, dialogue_history_list,
                                                dialogue_history_method, cen_decen_framework):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' or cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

    The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

    Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    Hence, the current state is {pg_state_list[-1]}, with the possible actions:
    {state_update_prompt}

    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
      '''

  return user_prompt_1

def input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, dialogue_history, response_total_list,
                                                                         pg_state_list, dialogue_history_list,
                                                                         dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''  
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in ('_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
    '''

  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_fast_plan_func(state_update_prompt_local_agent, state_update_prompt_other_agent,
                                                 dialogue_history, response_total_list, pg_state_list, dialogue_history_list,
                                                 dialogue_history_method, initial_plan=''):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
  Your response:
    '''
  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, dialogue_history, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method, initial_plan = ''):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
          '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
    
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}
  
    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}
  
    The initial plan is: {{{initial_plan}}}
    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
    Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
    Propose exactly one action for yourself at the **current** round.
    {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
    End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
    Your response:
    '''
  return user_prompt_1

def input_prompt_local_agent_HMAS2_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, central_response, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.

  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  {collision_avoidance_prompt} Please check the given plan, especially avoiding the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
          '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
    The current state is {pg_state_list[-1]}
    The central planner\'s current action plan is: {{{central_response}}}.
  
    {collision_avoidance_prompt} Please check the given plan, especially avoiding the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
    '''
  return user_prompt_1


def input_reprompt_func(state_update_prompt):
  user_reprompt = f'''
  Finished! The updated state is as follows(combined targets and boxes with the same color have been removed):

  {state_update_prompt}

  The output should be like json format like: {{Agent[0.5, 0.5]:move(box_blue, position[0.0, 1.0]), Agent[1.5, 0.5]:move...}}. If no action for one agent in the next step, just do not include its action in the output. Also remember at most one action for each agent in each step. {collision_avoidance_prompt}

  Next step output:
  '''
  return user_reprompt

def message_construct_func(user_prompt_list, response_total_list, dialogue_history_method):
  if f'{dialogue_history_method}' == '_w_all_dialogue_history':
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    #print('length of user_prompt_list', len(user_prompt_list))
    for i in range(len(user_prompt_list)):
      messages.append({"role": "user", "content": user_prompt_list[i]})
      if i < len(user_prompt_list)-1:
        messages.append({"role": "assistant", "content": response_total_list[i]})
    #print('Length of messages', len(messages))
  elif f'{dialogue_history_method}' in ('_wo_any_dialogue_history', '_w_only_state_action_history'):
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": user_prompt_list[-1]})
    #print('Length of messages', len(messages))
  return messages

def input_prompt_copy(state_update_prompt, pg_state_list, guidelines=''):

  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent operates in a 1x1 grid and can only interact with objects located at the corners of its square. Agents can move a box to one of the three other corners or to a same-color target within their square.
  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). 

  Your task is to instruct each agent to match all boxes to their same-color targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  [Action Output Instruction]
  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}.
  There may be multiple agents with possible actions that can move the same box, but you must choose only one. Different boxes cannot be moved to the same position.
  {guidelines}

  Reason about the task step-by-step, strictly follow [Action Output Instruction] to plan the next action:
    '''

  return user_prompt

def guidelines_prompt(state_update_prompt, pg_state, ref_prompt, box_prompt):

  user_prompt = f'''
  Now, the possible actions for each agent:
  {state_update_prompt}

  Your task is to generate guidelines for each agent to achieve the following goals:
  {box_prompt}
  {ref_prompt}

  If the agent has no suitable action, let it do nothing. Your guidelines in this json format:{{"Agent[0.5, 0.5]":"move box_blue to position[0.0, 2.0]", "Agent[0.5, 1.5]":"do nothing", "Agent[0.5, 2.5]":"move box_purple0 to target_purple"}}
  
  Based on [Reference knowledge] to generate your guidelines:
  '''

  return user_prompt

def input_prompt_feedback(state_update_prompt, response, feedback):
  feedback_prompt = f'''
  Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. The squares are identified by their center coordinates, e.g., square[0.5, 0.5].

  Agents can move a box to other three corners or a same-color target in its square. {collision_avoidance_prompt}

  Specify action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a next action.

  Now, each agent's possible actions are: {state_update_prompt}
  The current action plan is {response}

  But received feedback:
  {feedback} 

  Your task is to reflect on the reasons why the action plan was incorrect based on the feedback so as to avoid repeating these mistakes in future action replans. Make sure each reason within 128 token and put your answer in this format:'reason1:..., reason2:...'
  Your response:
    '''
  return feedback_prompt

def sequential_action_check_prompt(pg_dict_input, agent_actions_dict, cur_action_dict, feedback_error_prompt, guidelines):
  feedback_prompt = f'''
  Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
  Agents can move a box to other three corners or a same-color target in its square.
  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}.

  The current state is {pg_dict_input}
  The current action plan is {cur_action_dict}
  But received feedback:
  {feedback_error_prompt}

  The agent's possible actions are: 
  {agent_actions_dict}

  {guidelines}
  The action assigned to the agent must be selected from the possible actions. Different boxes cannot be moved to the same position.
  If multiple actions are moving the same box, only keep one. Example: {{'Agent[1.5, 3.5]': 'move(box_red0, position[2.0, 4.0])', 'Agent[1.5, 2.5]': 'move(box_red0, position[2.0, 2.0])'}}, only keep one of them, such as 'Agent[1.5, 3.5]': 'move(box_red0, position[2.0, 4.0])'.

  Your task is to replan actions based on the feedback information. Please replan again with the same ouput format:
  '''
  return feedback_prompt

def input_prompt_collision_feedback(pg_dict, response, feedback):
  feedback_prompt = f'''
  Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. The squares are identified by their center coordinates, e.g., square[0.5, 0.5].

  Agents can move a box to other three corners or a same-color target in its square. {collision_avoidance_prompt}

  Specify action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a next action.

  Now, the current state is: {pg_dict}
  The current action plan is {response}

  But received feedback:
  {feedback} 

  Your task is to reflect on the reasons why the action plan was incorrect based on the feedback so as to avoid repeating these mistakes in future action replans. Make sure each reason within 128 token and put your answer in this format:'reason1:..., reason2:...'
  Your response:
    '''
  return feedback_prompt

def input_prompt_target_belief(pg_state_list):
  
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. 
  
  Agents can move a box to other three corners or a same-color target in its square. The squares are identified by their center coordinates, e.g., square[0.5, 0.5].

  [output instruction]
  Your task is to match the box to the corner of the square where the target of the same color is located, in this json form: {{"box[id]":"position[coordinate]"}}, indicates that box[id] needs to move to position[coordinate]. **Note that position[coordinate] is the corners of the square.** All boxes must include. 
  Example: state: {{'0.5_0.5': ['target_blue'], '0.5_1.5': [], '1.5_0.5': [], '1.5_1.5': ['target_green'], '0.0_0.0': [], '0.0_1.0': [], '0.0_2.0': [], '1.0_0.0': [], '1.0_1.0': [], '1.0_2.0': ['box_blue'], '2.0_0.0': [], '2.0_1.0': [], '2.0_2.0': ['box_green']}}
  think: target_blue is at 0.5_0.5, the four corners of 0.5_0.5 are 0.0_0.0, 0.0_1.0, 1.0_0.0 and 1.0_1.0, and box_blue is at 1.0_2.0, so box_blue can match 1.0_1.0. target_green is at 1.5_1.5, the four corners of 1.5_1.5 are 1.0_1.0, 1.0_2.0, 2.0_1.0 and 2.0_2.0, and box_green is at 2.0_2.0, so box_green can directly match target_green.
  response: {{"box_blue": "position[1.0_1.0]", "box_green": "target_green"}}

  Now, the current state is {pg_state_list[-1]}.

  Reason about the task step-by-step, and strictly follow the [output instruction] to generate your response:
    '''

  return user_prompt_1

def input_ref_prompt(pg_dict, state_update_prompt, box_target_update, ref_action):
  user_prompt_1 = f'''
  Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. 
  
  Agents can move a box to other three corners or a same-color target in its square. {collision_avoidance_prompt}

  Now, each agent's possible actions are: {state_update_prompt}
  To match all boxes to a target of the same color, you might:
  {box_target_update}

  [Reference]
  {ref_action}

  [Reference] is some knowledge that may be used as a reference. Each box can be moved by which agent and where it will be moved next in order to match all boxes to the same-colored target. Each box can only be moved by one agent.
  You need to give advice. Make sure each advice within 256 token and put your answer in this format:'advice1:..., advice2:...'
    '''
  return user_prompt_1
# You cannot move the boxes to the same position, otherwise a collision will occur. For example, 'Agent[0.5, 1.5]': 'move(box_blue0, position[0.0, 2.0])' and 'Agent[0.5, 2.5]': 'move(box_red0, position[0.0, 2.0])' will collide. You should avoid this in your action plan.
def input_prompt_partial_agent_func(pg_dict, agents, agent_actions_prompt, cur_action_dict, guidelines):
  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent operates in a 1x1 grid and can only interact with objects located at the corners of its square. Agents can move a box to one of the three other corners or to a same-color target within their square.
  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). 

  Hence, the current state is {pg_dict}.
  
  Your task is to plan actions for the {agents}, aiming to move all boxes to the same color target. The actions should be in possible actions.
  with the possible actions:
  {agent_actions_prompt}

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}.
  The actions you plan cannot be the same as the position in the following actions:
  {cur_action_dict}

  [Output instruction]
  {guidelines}
  Before planning actions, consider that the currently assigned agents may not represent all available agents.

  Now, plan the next action plan for the {agents}:
    '''
  return user_prompt

def input_prompt_partial_agent_guidelines_func(pg_dict, agents, agent_actions_prompt, guidelines, task_desc, cur_action_dict):
  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent operates in a 1x1 grid and can only interact with objects located at the corners of its square. Agents can move a box to one of the three other corners or to a same-color target within their square.

  {task_desc}

  Plan actions for the {agents} to complete the [Task Description], with the possible actions:
  {agent_actions_prompt}

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, position[0.0, 2.0])", "Agent[1.5, 0.5]":"WAIT...}}.
  The actions you plan cannot be the same as the position in the following actions:
  {cur_action_dict}

  {guidelines}

  [Output instruction]
  **Before planning actions, consider that the currently assigned agents may not represent all available agents.**
  For example, Agent[0.5, 1.5] can move box_red0, but [Suggestion] "Move the box_red0 to the position[0.0, 3.0]" is invalid. Instead, consider having Agent[0.5, 1.5] WAIT; Agent[0.5, 3.5] can move box_blue0, but [Suggestion] "Move the box_blue0 directly to the target_blue" is invalid. Instead, consider having Agent[0.5, 3.5] WAIT.

  Now, strictly follow the [output instruction] to plan the next action plan for the {agents}:
'''
  return user_prompt