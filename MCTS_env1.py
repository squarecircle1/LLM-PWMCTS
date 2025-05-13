import json
import os
from tqdm import tqdm
from env1_create import *
import copy
import networkx as nx
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import itertools
import pprint
import sys

LAMBDA = 0.5

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def is_empty(state):
    count = 0
    for key, value in state.items():
        count += len(value)
    if count == 0:
        return 1
    return 0

def get_box_info(pg_dict):
    box_pos = {}
    for position, items in pg_dict.items():
        for item in items:
            if item.startswith('box_'):
                box_pos[item] = position
    return box_pos

def surround_index_func(row_num, coloum_num, row_index, coloum_index):
  surround_index_list = []
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    if i>=0 and i<=row_num-1 and j>=0 and j<=coloum_num-1 and not (i == row_index and j == coloum_index):
      surround_index_list.append([i+0.5,j+0.5])
  return surround_index_list

def surround_index_agent_func(row_num, coloum_num, row_index, coloum_index):
  surround_index_list = []
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    if i>=0 and i<=row_num and j>=0 and j<=coloum_num:
      surround_index_list.append([i,j])
  return surround_index_list

# 生成每个box的轨迹
def gen_box_trajectories(initial_state, action_list):
  box_trajectories = {}
  for position, items in initial_state.items():
      for item in items:
          if item.startswith('box_'):
              if item not in box_trajectories.keys():
                  box_trajectories[item] = {'positions': [], 'actions': [], 'agents':[]}
                  box_trajectories[item]['positions'].append(position)
              if position != box_trajectories[item]['positions'][-1]:
                box_trajectories[item]['positions'].append(position)
  for actions in action_list:
    for agent, action in actions.items():
      if action == "WAIT":
        continue
      box_name = action.split('(')[1].split(',')[0]
      if 'square' in action:
        coordinates = re.search(r'square[\(\[](.*?)[\)\]]', action).group(1)
        next_x, next_y = map(float, coordinates.split(', '))
        box_trajectories[box_name]['positions'].append(f'{next_x}_{next_y}')
      if 'target' in action:
        coordinates = re.search(r'Agent\[(.*?)\]', agent).group(1)
        next_x, next_y = map(float, coordinates.split(', '))
        box_trajectories[box_name]['positions'].append(f'{next_x}_{next_y}')
      box_trajectories[box_name]['actions'].append(action)
      box_trajectories[box_name]['agents'].append(agent)
  return box_trajectories

def database_update_func(box_traj, database_input):
  database = copy.deepcopy(database_input)
  for box, items in box_traj.items():
    positions = items['positions']
    actions = items['actions']
    agents = items['agents']
    for i, pos in enumerate(positions):
      for j in range(i+1, len(positions)):
        step = j - i
        state = f'{pos} to {positions[j]}'
        state_dict = {item['state']: item for item in database}
        found_dict = state_dict.get(state)
        if found_dict:
          if step < int(found_dict['step']):
            database.remove(found_dict)
            database.append({"state": state, "action":actions[i] , "step": str(step), 'agent':agents[i]})
        else:
          database.append({"state": state, "action":actions[i] , "step": str(step), 'agent':agents[i]})
  return database

def group_agents_by_value(agent_dict):
    # 使用 defaultdict 来存储相同 value 的智能体
    value_to_agents = defaultdict(list)
    # 遍历字典，将智能体按照 value 分组
    for agent, value in agent_dict.items():
        value_to_agents[value].append(agent)
    # # 将分组后的结果按照 value 升序排序
    # sorted_groups = sorted(value_to_agents.items(), key=lambda x: x[0])
    # 将分组后的结果按照 value_to_agents[value] 的长度降序排序
    sorted_groups = sorted(value_to_agents.items(), key=lambda x: len(x[1]), reverse=True)
    # 提取智能体列表
    result = [group[1] for group in sorted_groups]
    return result

def remove_agents_and_actions(agent_actions_dict, agents_to_remove):
    # 删除指定的智能体
    for agent in agents_to_remove:
        if agent in agent_actions_dict.keys():
            del agent_actions_dict[agent]

    return agent_actions_dict

def find_agents_for_boxes(agent_actions_dict):
    box_to_agents = {}
    # 遍历每个智能体的动作列表
    for agent, actions in agent_actions_dict.items():
        for action in actions:
            if action == 'WAIT':
                continue
            # 提取箱子的名称
            box = action.split('(')[1].split(',')[0]
            # 如果箱子不在字典中，初始化一个空列表
            if box not in box_to_agents:
                box_to_agents[box] = []
            # 如果智能体不在列表中，添加智能体
            if agent not in box_to_agents[box]:
                box_to_agents[box].append(agent)
    return box_to_agents

def parse_agent_position(agent_str):
    """从字符串中解析出智能体的坐标"""
    parts = agent_str.strip('Agent[]').split(', ')
    return float(parts[0]), float(parts[1])

def are_adjacent(agent1, agent2):
    """检查两个智能体是否相邻"""
    x1, y1 = parse_agent_position(agent1)
    x2, y2 = parse_agent_position(agent2)
    
    # 检查是否在同一行或同一列，并且距离为1
    return (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)

def get_conflict_graph(agents_actions):
    # 创建冲突图
    G = nx.Graph()
    
    # 添加每个agent为节点
    for agent in agents_actions:
        G.add_node(agent)
    
    # 遍历所有agent的动作
    for agent1, actions1 in agents_actions.items():
        for agent2, actions2 in agents_actions.items():
            # 不同的智能体之间进行比较，避免自比较
            if agent1 != agent2:
                # 查找是否有相同的square目标
                conflict = False
                for action1 in actions1:
                    for action2 in actions2:
                        # 如果两个动作都包含相同的square，说明这两个动作冲突
                        if 'square' in action1 and 'square' in action2:
                            square1 = action1.split('square')[1].split(']')[0]
                            square2 = action2.split('square')[1].split(']')[0]
                            if square1 == square2:
                                conflict = True
                                break
                    if conflict:
                        break
                
                # 如果有冲突，添加边
                if conflict:
                    G.add_edge(agent1, agent2)
    
    return G

class StateNode:
    def __init__(self, score=0, done=False):
        self.state = None
        # 父动作节点
        self.pre_action = None
        # 节点id
        self.id = None
        self.history = []
        # 上一个状态节点
        self.parent = None
        self.score = score
        self.done = done
        self.N = 0

        # 用于PUCT
        self.children = []
        self.children_probs = []
        self.agents_groups = None
        self.agents_groups_joint_actions = None
        self.agents_groups_actions = None
        self.agents_groups_score = None
        self.agents_groups_expand_actions = None
        self.agents_groups_expand_actions_probs = None

class ActionNode:
    def __init__(self, simulation=0, termination = False):
        self.action = None
        self.agent = None
        self.N = 0
        self.Q = 0
        self.Rs = []
        # 父动作节点
        self.pre_action = None
        # 子节点,可以是状态节点也可以是动作节点
        self.children = []
        self.children_probs = []
        self.agents_groups = None
        self.agents_groups_joint_actions = None
        self.agents_groups_actions = None
        self.agents_groups_score = None
        self.agents_groups_expand_actions = None
        self.agents_groups_expand_actions_probs = None

        self.children_id = None
        self.simulation = simulation
        self.termination = termination

class MCTS_env:
    def __init__(self, pg_dict, row_num, column_num, model_name):
        super().__init__()
        self.state = pg_dict
        self.init_state = copy.deepcopy(self.state)
        self.history = []
        self.pg_row_num = row_num
        self.pg_column_num = column_num
        self.model_name = model_name
        box_pos = get_box_info(pg_dict)
        self.box_num = len(box_pos)
        # 记录每次simulation中的数据
        self.prompt_list = []
        self.sim_action_history = []
        self.sim_state_history = []
        self.widden_info_list = []
        self.translation_lm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def reset(self, pg_dict):
        self.state = pg_dict
        self.init_state = copy.deepcopy(self.state)
        self.history = []
        # 重置记录每次simulation数据的变量
        self.prompt_list = []
        self.sim_action_history = []
        self.sim_state_history = []
        self.widden_info_list = []

    def step(self, action):
        self.history.append(action)
        _, next_state, _=action_from_response_collision(self.state, action)
        old_box_pos = get_box_info(self.state)
        new_box_pos = get_box_info(next_state)
        self.state = next_state
        # reward = (len(old_box_pos) - len(new_box_pos))/self.box_num
        # 如果没有box了, 说明任务结束
        count = 0
        for key, value in next_state.items():
            count += len(value)
        if count == 0:
            done = True
            reward = 1
        else:
            done = False
            reward = 0

        return next_state, reward, done, self.history

    # 在当前state, 随机选择一个action
    def random_action(self, pg_dict):
        pg_dict_copy = copy.deepcopy(pg_dict)
        response_dict = {}
        for i in range(self.pg_row_num):
            for j in range(self.pg_column_num):
                agent = f'Agent[{i+0.5}, {j+0.5}]'
                square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
                square_item_only_box = [item for item in square_item_list if item[:3]=='box']
                surround_index_list = surround_index_func(self.pg_row_num, self.pg_column_num, i, j)
                action_list = []
                for box in square_item_only_box:
                    for surround_index in surround_index_list:
                        action_list.append(f'move({box}, square{surround_index})')
                    if 'target'+box[3:-1] in square_item_list:
                        action_list.append(f'move({box}, target{box[3:-1]})')
                if len(action_list):
                    random_number = random.randrange(len(action_list))
                    action = action_list[random_number]
                    response_dict[agent] = action
        return response_dict

    # 得到状态pg_dict下所有agent的动作列表
    def get_all_agents_actions_dict_func(self, pg_dict):
        pg_dict_copy = copy.deepcopy(pg_dict)
        agents_actions = {}
        for i in range(self.pg_row_num):
            for j in range(self.pg_column_num):
              square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
              square_item_only_box = [item for item in square_item_list if item[:3]=='box']
              surround_index_list = surround_index_func(self.pg_row_num, self.pg_column_num, i, j)
              action_list = []
              for box in square_item_only_box:
                for surround_index in surround_index_list:
                  action_list.append(f'move({box}, square{surround_index})')
                if 'target'+box[3:-1] in square_item_list:
                  action_list.append(f'move({box}, target{box[3:-1]})')
                if action_list:
                  agents_actions[f'Agent[{i+0.5}, {j+0.5}]'] = action_list
        return agents_actions

    def is_termination(self, pg_dict, assigned_agents, assigned_actions):
        agents_actions = self.get_all_agents_actions_dict_func(pg_dict)
        boxes_to_remove = []
        for _, action in assigned_actions.items():
            match = re.search(r'move\((.*?),', action)
            if match:
                box = match.group(1)
                boxes_to_remove.append(box)

        # 移除了已分配的智能体和已分配的动作，新的可行动作
        agents_actions = remove_agents_and_actions(agents_actions, assigned_agents)

        G = get_conflict_graph(agents_actions)
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        agents_groups = group_agents_by_value(coloring)

        filter_agents_group = []
        for agents_group in agents_groups:
            agents_actions = self.get_agents_group_actions_func(pg_dict, agents_group, assigned_actions)
            agents_group_copy = []
            for agent in agents_group:
                if agent in agents_actions.keys():
                    agents_group_copy.append(agent)
            if agents_group_copy:
                filter_agents_group.append(agents_group_copy)

        if filter_agents_group:
            return False
        else:
            return True

    def get_agents_group_actions_func(self, pg_dict, next_agents, assigned_actions):
        pg_dict_copy = copy.deepcopy(pg_dict)
        box_pos = get_box_info(pg_dict)
        for action in assigned_actions.values():
            box = re.search(r'move\((.*?),', action).group(1)
            if 'square' in action:
                coordinates = re.search(r'square[\(\[](.*?)[\)\]]', action).group(1)
                next_x, next_y = map(float, coordinates.split(', '))
                box_pos[box] = f'{next_x}_{next_y}'
            else:
                box_pos[box] = 'target'
        pprint.pprint(box_pos)
        agents_actions = {}
        for agent in next_agents:
            action_list = []
            agent_x, agent_y = parse_agent_position(agent)
            square_item_list = pg_dict_copy[str(agent_x)+'_'+str(agent_y)]
            square_item_only_box = [item for item in square_item_list if item[:3]=='box']
            if square_item_only_box:
                box = square_item_only_box[0]
                for x,y in surround_index_agent_func(self.pg_row_num, self.pg_column_num, agent_x, agent_y):
                    if f'{x}_{y}' not in box_pos.values():
                        action_list.append(f'move({box}, square({x}, {y}))')
                if 'target'+box[3:-1] in square_item_list:
                    action_list.append(f'move({box}, target{box[3:-1]})')
            if action_list:
                agents_actions[agent] = action_list
        return agents_actions

    # 将未分配的agent划分为多个组, 并且得到每个组的所有可行联合动作
    def get_agents_groups(self, pg_dict, assigned_agents=[], assigned_actions={}):
        agents_actions = self.get_all_agents_actions_dict_func(pg_dict)
        agents_actions = remove_agents_and_actions(agents_actions, assigned_agents)
        pprint.pprint(agents_actions)

        G = get_conflict_graph(agents_actions)
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        agents_groups = group_agents_by_value(coloring)

        print(agents_groups)

        agents_groups_joint_actions = []
        agents_groups_actions = []
        filter_agents_group = []
        for agents_group in agents_groups:
            agents_actions = self.get_agents_group_actions_func(pg_dict, agents_group, assigned_actions)
            pprint.pprint(agents_actions)
            agents_group_copy = []
            for agent in agents_group:
                if agent in agents_actions.keys():
                    agents_group_copy.append(agent)
            if agents_group_copy:
                filter_agents_group.append(agents_group_copy)
                agents_groups_actions.append(agents_actions)
                agents_joint_actions = self.get_joint_actions_list(agents_actions)
                agents_groups_joint_actions.append(agents_joint_actions)

        return filter_agents_group, agents_groups_joint_actions, agents_groups_actions
    
    def get_joint_actions_list(self, agents_actions_dict):
        combinations = list(itertools.product(*agents_actions_dict.values()))
        # 将每个组合转换为字典列表
        joint_actions = []
        for combination in combinations:
            joint_action = dict(zip(agents_actions_dict.keys(), combination))
            joint_actions.append(joint_action)
        return joint_actions
    
    def get_valid_actions(self, pg_dict, assigned_agents_input, assigned_agents_actions_input, agents_group_input, agents_group_expand_actions, agents_group_joint_actions, agents_group_actions, guidelines):
        assigned_agents = assigned_agents_input.copy()
        assigned_agents_actions = assigned_agents_actions_input.copy()
        agents_group = agents_group_input.copy()
        # # agents_group的所有可行的联合动作
        # agents_group_actions = self.get_agents_group_actions_func(pg_dict, agents_group, assigned_agents_actions)
        # joint_actions = self.get_joint_actions_list(agents_group_actions)
        agents_group_actions = agents_group_actions.copy()
        joint_actions = agents_group_joint_actions.copy()

        agents_group_actions_prompt = ''
        for agent in agents_group:
            floats = re.findall(r'\d+\.\d+', agent)
            x, y = tuple(float(f) for f in floats)
            agents_group_actions_prompt += f'{agent}: I am in square[{x}, {y}], I can do {agents_group_actions[agent]}\n'

        user_prompt = input_prompt_partial_agent_func(pg_dict, agents_group, agents_group_actions_prompt, assigned_agents_actions, guidelines)

        messages = message_construct_func([user_prompt], [], '_w_all_dialogue_history')
        response, token_num_count = GPT_response_1(messages, self.model_name)
        match = re.search(r'{.*}', response, re.DOTALL)
        if match:
            response = match.group()

        joint_actions_probs = self.calculate_emperical_prob(response, joint_actions)

        # 从为扩展的联合动作中，选择概率最大的加入进来, 然后expand_actions是按照顺序添加进来的已扩展动作列表
        for action in joint_actions:
            if action in agents_group_expand_actions:
                joint_actions_probs[joint_actions.index(action)] = 0
        idx = np.argmax(joint_actions_probs)
        expand_action = joint_actions[idx]
        expand_action_prob = joint_actions_probs[idx]

        assigned_agents_actions.update(expand_action)
        assigned_agents += agents_group

        return assigned_agents, assigned_agents_actions, expand_action, expand_action_prob

    def _find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, show_progress_bar=False)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        cos_scores = cos_scores - np.mean(cos_scores) 
        return cos_scores
    
    def calculate_emperical_prob(self, response, joint_actions):
        # joint_actions是dict的list，要把每个dict先转化为string
        string_joint_actions = [json.dumps(d) for d in joint_actions]
        valid_action_embedding = self.translation_lm.encode(string_joint_actions, convert_to_tensor=True, show_progress_bar=False)
        actions_dis = np.zeros(len(joint_actions))
        cos_sim = self._find_most_similar(response, valid_action_embedding)
        actions_dis = np.exp(10 * cos_sim) / np.sum(np.exp(10 * cos_sim), axis=0)
        return actions_dis


class MCTSAgent:
    def __init__(self, env, max_depth, simulation_num, uct_type='PUCT', search_engine = None, use_llm = True, use_b=True):
        self.uct_type = uct_type
        self.root = None
        self.exploration_constant = 24
        self.max_depth = max_depth
        self.discount_factor = 0.95
        self.simulation_num = simulation_num
        self.simulation = 0
        self.use_llm = use_llm
        # 渐进扩展，是否使用database
        self.use_b = use_b
        # env类
        self.env = env
        # id --> state_node action_node
        self.state_dict = {}
        # 控制processing widden的参数
        self.k = 1
        self.a = 0.3
        # database检索
        self.search_engine = search_engine
        self.database = []
        self.belief_dict = {}
    
    @staticmethod
    def state_id(history: list):
        # history应该是从根节点到当前节点的action列表，但是action是dict类型
        history_str_list = [json.dumps(item) for item in history]
        return ' '.join(history_str_list)

    # 构建state node
    def build_state(self, pg_dict, history, done, score=0, pre_action = None):
        state = StateNode()
        state.state = pg_dict
        state.done = done
        state.score = score
        state.pre_action = pre_action
        state.history = history
        state.id = self.state_id(history)
        self.state_dict[state.id] = state

        # 每个state下根据冲突图分为多个group
        state.agents_groups, state.agents_groups_joint_actions, state.agents_groups_actions = self.env.get_agents_groups(pg_dict)
        state.agents_groups_score = np.zeros(len(state.agents_groups))
        state.agents_groups_expand_actions = [[] for _ in range(len(state.agents_groups))]
        state.agents_groups_expand_actions_probs = [[] for _ in range(len(state.agents_groups))]

        return state

    def build_action(self, pg_dict, assigned_action, assigned_agent, simulation, termination):
        action = ActionNode()
        action.agent = assigned_agent
        action.action = assigned_action
        action.termination = termination
        action.simulation = simulation
        # action下进行分组
        if not termination:
            action.agents_groups, action.agents_groups_joint_actions, action.agents_groups_actions = self.env.get_agents_groups(pg_dict, assigned_agent, assigned_action)
            action.agents_groups_score = np.zeros(len(action.agents_groups))
            action.agents_groups_expand_actions = [[] for _ in range(len(action.agents_groups))]
            action.agents_groups_expand_actions_probs = [[] for _ in range(len(action.agents_groups))]

        return action

    def gen_belief_dict(self, pg_dict):
        target_belief_prompt = input_prompt_target_belief([pg_dict])
        messages = message_construct_func([target_belief_prompt], [], '_w_all_dialogue_history') # message construction
        belief_response, token_num_count = GPT_response(messages, self.env.model_name)
        print(belief_response)
        match = re.search(r'{.*}', belief_response, re.DOTALL)
        if match:
            belief_response = match.group()
            belief_response_dict = json.loads(belief_response)
        belief_dict = {}
        for key, value in belief_response_dict.items():
            square = tuple(map(float, re.findall(r"\d+\.?\d*", value)))
            if square:
                belief_dict[key] = str(square[0])+'_'+str(square[1])
            else:
                belief_dict[key] = value
        return belief_dict
    
    def get_guidlines_fun(self, pg_dict):
        if not self.search_engine.is_load:
            return None, 0, None
        # 当前状态，存在的box以及所处位置
        box_pos_dict = get_box_info(pg_dict)
        # guidelines = '[Reference knowledge]'
        guidelines = '[Suggestion]'
        task_desc = "[Task description]"
        score = 0
        num = 0
        for box, box_pos in box_pos_dict.items():
            box_target_pos = self.belief_dict[box]
            if box_pos == box_target_pos or box_target_pos.startswith('target'):
                # guidelines += f'\nMove the {box} directly to the target.'
                guidelines += f'\nmove({box}, target_{box[4:-1]})'
                task_desc += f'\nThe {box} needs to move to the same-color target.'
                continue
            query = f'{box_pos} to {box_target_pos}'
            res = self.search_engine.search_by_query_scores(query)
            ref_action = res[0][0].metadata['action']
            print('-------ref_action--------')
            print(ref_action)
            coordinates = re.search(r'square[\(\[](.*?)[\)\]]', ref_action).group(1)
            next_x, next_y = map(float, coordinates.split(', '))
            ref_state = res[0][0].page_content
            # guidelines += f'\nMove the {box} to the position[{next_x}, {next_y}].'
            guidelines += f'\nmove({box}, square({next_x}, {next_y}))'
            task_desc += f'\nThe {box} need from {box_pos} to {box_target_pos}'
            score += res[0][1]
            num += 1
        if num:
            score = score/num
        else:
            score = 1
        return guidelines, score, task_desc

    def get_agents_groups_score(self, agents_groups, agents_groups_actions, guidelines):
        if guidelines is None:
            return np.ones(len(agents_groups))
        # print('----------agents_groups_actions--------')
        # pprint.pprint(agents_groups_actions)
        # print('----------agents_group--------')
        # pprint.pprint(agents_groups)
        guidelines_actions_list = guidelines.strip().split('\n')[1:]
        agents_group_score_list = []
        for group_i, agents_group in enumerate(agents_groups):
            agents_group_score = 0
            for agent in agents_group:
                agent_actions = agents_groups_actions[group_i][agent]
                agent_actions_embedding = self.env.translation_lm.encode(agent_actions, convert_to_tensor=True, show_progress_bar=False)
                agent_score = 0
                for guidelines_action in guidelines_actions_list:
                    guidelines_action_embedding = self.env.translation_lm.encode(guidelines_action, convert_to_tensor=True, show_progress_bar=False)
                    cos_scores = st_utils.pytorch_cos_sim(guidelines_action_embedding, agent_actions_embedding)[0].detach().cpu().numpy()
                    max_cos_score = np.max(cos_scores)
                    if 'target' in guidelines_action:
                        agent_score += max_cos_score
                    else:
                        agent_score += max_cos_score
                    # if guidelines_action in agent_actions and 'target' in guidelines_action:
                    #     agent_score += 2
                    # elif guidelines_action in agent_actions:
                    #     agent_score += 1
                agent_score /= len(guidelines_actions_list)
                agents_group_score += agent_score
            agents_group_score_list.append(agents_group_score/len(agents_group))
        return np.array(agents_group_score_list)

    def update_children_probs(self, node):
        children_probs = np.zeros(len(node.children))
        for child_i, child in enumerate(node.children):
            action = child.action
            for i, expand_actions in enumerate(node.agents_groups_expand_actions):
                if action in expand_actions:
                    agents_group_score = node.agents_groups_score[i]
                    action_prob = node.agents_groups_expand_actions_probs[i][expand_actions.index(action)]
                    children_probs[child_i] = agents_group_score*action_prob
                    break
        children_probs = softmax(children_probs)
        children_probs = LAMBDA * children_probs + (1-LAMBDA) / len(node.children)
        return children_probs
    
    def greedy_action_node(self, node, exploration_constant, if_print=False):
        best_value = -np.inf
        best_children = []
        for i in range(len(node.children)):
            child = node.children[i]
            child_prob = node.children_probs[i]

            if exploration_constant == 0:
                ucb_value = child.Q
            elif self.uct_type == 'UCT':
                ucb_value = child.Q + exploration_constant * np.sqrt(np.log(node.N + 1) / (child.N + 1))
                # print(child.Q, exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1)))
            elif self.uct_type == 'PUCT':
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(node.N) / (child.N + 1)
            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(i)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [i]
        if if_print:
            for c in node.children:
                if c.N > 0:
                    print(c.action, c.Q, c.N)
        return best_children[0]
    
    def select_expand_action_node(self, state_node):
        assigned_agents = []
        assigned_agents_actions = {}
        guidelines, _ , task_desc = self.get_guidlines_fun(state_node.state)
        print(guidelines)
        # 看是否满足渐进加宽的条件
        if len(state_node.children) <= self.k*state_node.N**self.a:
            agents_groups_score = self.get_agents_groups_score(state_node.agents_groups, state_node.agents_groups_actions, guidelines=None)
            state_node.agents_groups_score = agents_groups_score

            # test
            print('state_node.state: ', state_node.state)
            print('state_node.agents_groups: ', state_node.agents_groups)
            print('state_node.agents_groups_actions: ')
            pprint.pprint(state_node.agents_groups_actions)

            best_agents_group = state_node.agents_groups[np.argmax(agents_groups_score)]
            best_agents_group_expand_actions = state_node.agents_groups_expand_actions[np.argmax(agents_groups_score)]
            best_agents_group_actions = state_node.agents_groups_actions[np.argmax(agents_groups_score)]
            best_agents_group_joint_actions = state_node.agents_groups_joint_actions[np.argmax(agents_groups_score)]

            if len(best_agents_group_expand_actions) < len(best_agents_group_joint_actions):
                assigned_agents, assigned_agents_actions, expand_action, expand_action_prob= self.env.get_valid_actions(state_node.state, assigned_agents, assigned_agents_actions, best_agents_group, best_agents_group_expand_actions, best_agents_group_joint_actions, best_agents_group_actions, guidelines)
                termination = self.env.is_termination(state_node.state, assigned_agents, assigned_agents_actions)
                state_node.children.append(self.build_action(state_node.state, assigned_agents_actions, assigned_agents, self.simulation, termination))
                state_node.agents_groups_expand_actions[np.argmax(agents_groups_score)].append(expand_action)
                state_node.agents_groups_expand_actions_probs[np.argmax(agents_groups_score)].append(expand_action_prob)
                state_node.children_probs = self.update_children_probs(state_node)

        # 从状态节点选择下一个动作节点
        best_action_node_idx = self.greedy_action_node(state_node, self.exploration_constant)
        action_node = state_node.children[best_action_node_idx]

        # 说明还有agent没有分配完，没有到当前状态的终止动作节点
        while not action_node.termination:
            if len(action_node.children) <= self.k*action_node.N**self.a:
                agents_groups_score = self.get_agents_groups_score(action_node.agents_groups, action_node.agents_groups_actions, guidelines=None)
                action_node.agents_groups_score = agents_groups_score

                # test
                print('state_node.state: ', state_node.state)
                print('action_node.agents_groups: ', action_node.agents_groups)
                print('action_node.agents_groups_actions: ')
                pprint.pprint(action_node.agents_groups_actions)

                best_agents_group = action_node.agents_groups[np.argmax(agents_groups_score)]
                best_agents_group_expand_actions = action_node.agents_groups_expand_actions[np.argmax(agents_groups_score)]
                best_agents_group_actions = action_node.agents_groups_actions[np.argmax(agents_groups_score)]
                best_agents_group_joint_actions = action_node.agents_groups_joint_actions[np.argmax(agents_groups_score)]

                if len(best_agents_group_expand_actions) < len(best_agents_group_joint_actions):
                    assigned_agents, assigned_agents_actions, expand_action, expand_action_prob = self.env.get_valid_actions(state_node.state, action_node.agent, action_node.action, best_agents_group, best_agents_group_expand_actions, best_agents_group_joint_actions, best_agents_group_actions, guidelines)
                    termination = self.env.is_termination(state_node.state, assigned_agents, assigned_agents_actions)

                    action_node.children.append(self.build_action(state_node.state, assigned_agents_actions, assigned_agents, self.simulation, termination))
                    action_node.agents_groups_expand_actions[np.argmax(agents_groups_score)].append(expand_action)
                    action_node.agents_groups_expand_actions_probs[np.argmax(agents_groups_score)].append(expand_action_prob)
                    action_node.children_probs = self.update_children_probs(action_node)

            best_action_node_idx = self.greedy_action_node(action_node, self.exploration_constant)
            next_action_node = action_node.children[best_action_node_idx]
            next_action_node.pre_action = action_node
            action_node = next_action_node

        # 已经到了终止动作节点，说明已经形成了完整的联合动作
        return action_node
    
    def search(self, pg_dict, history, done, Saving_path_result, index_query_times):
        self.state_dict = {}
        init_history = history.copy()
        # 创建保存当前搜索过程数据的文件夹
        Search_result = Saving_path_result+f'/search{index_query_times+1}'
        os.makedirs(Search_result, exist_ok=True)
        # root状态节点，history为[], done为false，同时会把root状态节点加入到state_dict中
        self.root = self.build_state(pg_dict, init_history, done)
        # 开始模拟
        for i in tqdm(range(self.simulation_num)):
            # 创建保存当前搜索中，当前simulate数据的文件夹
            self.simulation = i
            Search_sim_result = Search_result + f'/simulation{i+1}'
            os.makedirs(Search_sim_result, exist_ok=True)
            os.makedirs(Search_sim_result + f'/state', exist_ok=True)
            os.makedirs(Search_sim_result + f'/action', exist_ok=True)
            os.makedirs(Search_sim_result + f'/prompt', exist_ok=True)
            # 重置mcts的env
            self.env.reset(pg_dict)

            # 加载database的检索引擎
            if os.path.exists(Search_result + f'/simulation{i}/database.json'):
                self.search_engine.load_documents(Search_result + f'/simulation{i}/database.json')
            # 进行一轮simulation
            root = self.simulate()
            self.root = root
            # 记录simulation的数据
            for i, state in enumerate(self.env.sim_state_history):
                with open(Search_sim_result + f'/state' + f'/state{i+1}.json', 'w') as f:
                    json.dump(state, f)
            for i, action in enumerate(self.env.sim_action_history):
                with open(Search_sim_result + f'/action' + f'/action{i+1}.json', 'w') as f:
                    json.dump(action, f)
            for i, prompt in enumerate(self.env.prompt_list):
                with open(Search_sim_result +'/prompt' + '/prompt'+str(i+1), 'w') as f:
                    f.write(prompt)
            # 更新database
            box_traj = gen_box_trajectories(pg_dict, self.env.history)
            self.database = database_update_func(box_traj, self.database)
            with open(Search_sim_result + '/database.json', 'w') as f:
                json.dump(self.database, f)
            self.save_tree(Search_sim_result+'/mcts_tree.json')

        # 从状态节点选择下一个动作节点
        best_action_node_idx = self.greedy_action_node(self.root, 0, if_print=True)
        best_action_node = self.root.children[best_action_node_idx]
        # 说明还有agent没有分配完，没有到当前状态的终止动作节点
        while not best_action_node.termination:
            best_action_node_idx = self.greedy_action_node(best_action_node, 0, if_print=True)
            best_action_node = best_action_node.children[best_action_node_idx]
        return best_action_node.action

    def backpropagate(self, state_node, state_reward_dict, R):
        pre_state_node = state_node.parent
        while pre_state_node != None:
            reward = state_reward_dict[pre_state_node.id]
            action_node = state_node.pre_action
            # action_node.children = state_node
            while action_node != None:
                action_node.N += 1
                action_node.Rs.append(reward+R*self.discount_factor)
                action_node.Q = np.sum(np.array(action_node.Rs) * softmax(action_node.Rs, T=10))
                action_node = action_node.pre_action
            pre_state_node.N += 1
            state_node = pre_state_node
            pre_state_node = state_node.parent
            R = R * self.discount_factor

    def simulate(self):
        state_node = self.root
        depth = 0
        rollout_next = False
        state_reward_dict = {}
        self.env.sim_state_history.append(state_node.state)

        # 选择阶段
        print(f"------------------simulate{self.simulation+1}-----------------")
        best_action_node = self.select_expand_action_node(state_node)

        pg_dict, reward, done, history = self.env.step(best_action_node.action)
        state_reward_dict[state_node.id] = reward
        self.env.sim_action_history.append(best_action_node.action)
        print('---------simulate action--------')
        pprint.pprint(best_action_node.action)

        next_state_id = self.state_id(history)
        # 看下一个状态是否之前被访问过
        while next_state_id in self.state_dict.keys():
            # 下一个状态节点
            next_state_node = best_action_node.children[0]
            self.env.sim_state_history.append(next_state_node.state)
            depth += 1
            if next_state_node.done or depth == self.max_depth:
                return self.root

            # 继续选择下一个状态节点的下一个动作节点
            best_action_node = self.select_expand_action_node(next_state_node)
            pg_dict, reward, done, history = self.env.step(best_action_node.action)
            state_reward_dict[next_state_node.id] = reward
            self.env.sim_action_history.append(best_action_node.action)

            print('---------simulate action--------')
            pprint.pprint(best_action_node.action)

            next_state_id = self.state_id(history)
            state_node = next_state_node

        # 说明到了没有访问过的state
        next_state_node = self.build_state(pg_dict, history, done, pre_action=best_action_node)
        next_state_node.parent = state_node

        best_action_node.children.append(next_state_node)
        rollout_next = True

        if rollout_next:
            if self.use_llm:
                rollout_r = []
                for _ in range(1):
                    random_r = self.rollout_llm(next_state_node.state, depth+1)
                    rollout_r.append(random_r)
                R = sum(rollout_r)/len(rollout_r)
            else:
                rollout_r = []
                for _ in range(1):
                    random_r = self.rollout(next_state_node.state, depth+1)
                    rollout_r.append(random_r)
                R = sum(rollout_r)/len(rollout_r)
        # R是rollout评估的状态节点的value
        self.backpropagate(next_state_node, state_reward_dict, R)

        return self.root
    
    def rollout(self, state, depth):
        if is_empty(state):
            return 1
        if depth == self.max_depth:
            boxes_all_list = [item for items in self.root.state.values() for item in items if item.startswith('box')]
            boxes_remaining_list = [item for items in state.values() for item in items if item.startswith('box')]
            lifted_weight_ratio = (len(boxes_all_list) - len(boxes_remaining_list)) / len(boxes_all_list)
            return lifted_weight_ratio
        # 在state处，采用随机策略进行rollout
        action = self.env.random_action(state)
        next_state, reward, done, history = self.env.step(action)
        if done:
            print("Done!")
        r = reward + self.discount_factor * self.rollout(next_state, depth+1)
        return r 
    
    def save_tree(self, filename):
        # 递归地遍历树并将其转换为 JSON 格式
        def serialize_node(node):
            if isinstance(node, StateNode):
                return {
                    'type': 'state',
                    'state': json.dumps(node.state),
                    'N':node.N,
                    'score':node.score,
                    'children': [serialize_node(child) for child in node.children] if node.children else {}
                }
            elif isinstance(node, ActionNode):
                return {
                    'type': 'action',
                    'action': json.dumps(node.action),
                    'N':node.N,
                    'Q':node.Q,
                    'Rs':node.Rs,
                    'simulation': node.simulation,
                    'children': [serialize_node(child) for child in node.children] if node.children else {}
                }
            else:
                raise ValueError("Unknown node type")

        # 将根节点序列化并保存到文件
        with open(filename, 'w') as f:
            json.dump(serialize_node(self.root), f, indent=4)