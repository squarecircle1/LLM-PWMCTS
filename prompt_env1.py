from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000

collision_avoidance_prompt = '[Remember that each square can only hold at most one box! Therefore, you need to avoid box collisions. Moving two boxes to the same square at the same time or moving a box to a square that already has a box is not allowed!]'

def LLM_summarize_func(state_action_prompt_next_initial, model_name):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name)
  return response


def input_prompt_1_func(state_update_prompt):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
  '''
  return user_prompt_1


def input_prompt_1_only_state_action_func(state_update_prompt, response_total_list, pg_state_list):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
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
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.
  
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
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
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and one box.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their same-color targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
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
    You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

    The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

    Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    Hence, the current state is {pg_state_list[-1]}, with the possible actions:
    {state_update_prompt}

    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

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
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
    All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target.
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    
    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}
    
    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}
    
    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
    Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
    Propose exactly one action for yourself at the **current** round.
    End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction]!
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
  one extra planner first proposes an initial plan to coordinates all agents to achieve the goal: match each box with its color-coded target.
  Then all the action agents discuss and coordiante with each other to come out a final plan.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
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
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
    one extra planner first proposes an initial plan to coordinates all agents to achieve the goal: match each box with its color-coded target.
    Then all the action agents discuss and coordiante with each other to come out a final plan.
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

    The initial plan is: {{{initial_plan}}}
    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and one box.
  one extra planner first proposes an initial plan to coordinates all agents to achieve the goal: match each box with its same-color target.
  Then all the action agents discuss and coordiante with each other to come out a final plan.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  {collision_avoidance_prompt} Check the given plan, especially to avoid collisions between the box you are moving and other boxes in the square. Avoid situations where two boxes move to the same square in the same step.
  Propose exactly one action for yourself at the **current** round.
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
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
    one extra planner first proposes an initial plan to coordinates all agents to achieve the goal: match each box with its color-coded target.
    Then all the action agents discuss and coordiante with each other to come out a final plan.
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

    The initial plan is: {{{initial_plan}}}
    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
    Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
    Propose exactly one action for yourself at the **current** round.
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.

  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.

  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The central planner\'s current action plan is: {{{central_response}}}.
  {collision_avoidance_prompt} Check the given plan, especially to avoid collisions between the box you are moving and other boxes in the square. Avoid situations where two boxes move to the same square in the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
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
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
  
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
    The central planner\'s current action plan is: {{{central_response}}}.
  
    Please evaluate the given plan. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
    '''
  return user_prompt_1


def input_reprompt_func(state_update_prompt):
  user_reprompt = f'''
  Finished! The updated state is as follows(combined targets and boxes with the same color have been removed):

  {state_update_prompt}

  The output should be like json format like: {{Agent[0.5, 0.5]:move(box_blue, square[0.5, 1.5]), Agent[1.5, 0.5]:move...}}. If no action for one agent in the next step, just do not include its action in the output. Also remember at most one action for each agent in each step.

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

def input_prompt_evolution(state_update_prompt, last_pg_dict, last_response):

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a next action; otherwise, exclude it from the action plan.
  
  Now, the current state is {last_pg_dict}, with the possible actions:
  {state_update_prompt}

  the current action plan is {last_response}

  If you agree with the current action, answer "pass". 
  If you think the action plan can be optimized, for instance, by assigning reasonable actions to agents currently without tasks to promote task completion, answer with the reason in the format: "Optimization reason:"
  Your answer:
    '''
  return user_prompt_1 

def input_prompt_feedback(state_update_prompt, response, feedback):
  feedback_prompt = f'''
  Each agent is assigned to a 1x1 square and can only interact with objects in its area. 

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Each square can contain many targets and boxes.

  Agents can move a box to a neighboring square or a same-color target in its square. 

  Specify action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a next action.

  Now, each agent's possible actions are: {state_update_prompt}
  The current action plan is {response}

  But received feedback:
  {feedback} 

  Your task is to reflect on the reasons why the action plan was incorrect based on the feedback so as to avoid repeating these mistakes in future action plans. Make sure each reason within 128 token and put your answer in this format:'reason1:..., reason2:...'
  Your response:
    '''
  return feedback_prompt

def input_prompt_copy(state_update_prompt, pg_state_list, response_total_list, guidelines):
  if response_total_list:
    previous_state = pg_state_list[-2]
    previous_action = response_total_list[-1]
  else:
    previous_state = ''
    previous_action = ''
  
  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target within the current square. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their same-color targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.
  
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  {guidelines}

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a action next. Now, plan the next step:
    '''

  return user_prompt

def input_prompt_partial_agent_func(pg_dict, agents, agent_actions_prompt, cur_action_dict, guidelines):
  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target within the current square. Each square can contain many targets and one box.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Hence, the current state is {pg_dict}.
  
  Your task is to plan actions for the {agents}, aiming to move all boxes to the same color target. The actions should be in possible actions.
  with the possible actions:
  {agent_actions_prompt}

  Specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue0, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
  The actions you plan cannot be the same as the position in the following actions:
  {cur_action_dict}

  [Output instruction]
  {guidelines}
  {collision_avoidance_prompt}

  Now, plan the next action plan for the {agents}:
    '''
  return user_prompt

def input_prompt_target_belief(pg_state_list):
  
  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. The boxes are identified by their color and id, e.g., box_blue0.

  [output instruction]
  Your task is to match all the boxes with the target, in this json form: {{"box[id]":"square[coordinate]:target"}}, indicates that the box[id] needs to match the target under the square. The box and target must be the same color. All boxes must include. Each target can only match one box. 
  Example: state: {{'0.5_0.5': ['box_blue0'], '0.5_1.5': [], '1.5_0.5': ['target_blue', 'target_green'], '1.5_1.5': ['box_green0']}}
  response: {{"box_blue0": "square[1.5_0.5]:target_blue", "box_green0": "square[1.5_0.5]:target_green"}}

  Now, the current state is {pg_state_list[-1]}. Strictly follow the [output instruction] to generate your response:
    '''

  return user_prompt

def input_prompt_react(state_update_prompt, pg_state_list):

  user_prompt = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target in its area square. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their same-color targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  [Output Instruction]
  Think step-by-step about the task, before you output your action plan, output your thought in this format: 'think: ...'.
  Then output your action plan, specify your action plan in this json format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a action next. 
  If the box is to move to the target, the color must be the same. Each agent can only be assigned one action.

  Example#:  The current state is {{'0.5_0.5': [], '0.5_1.5': ['target_blue'], '1.5_0.5': ['target_red', 'box_red0'], '1.5_1.5': ['box_blue0']}}
  Think: The box_red0 and target_red are both in 1.5_0.5, so Agent[1.5, 0.5] can move box_red0 to target_red. The box_blue0 is at 1.5_1.5, target_blue is at 0.5_1.5, so Agent[1.5, 1.5] can move box_blue0 to 0.5_1.5
  Action: {{"Agent[1.5, 0.5]": "move(box_red0, target_red)", "Agent[1.5, 1.5]": "move(box_blue0, square[0.5, 1.5])"}}

  The current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Now, strictly follow [Output Instruction] to format and output the plan:
    '''
  return user_prompt