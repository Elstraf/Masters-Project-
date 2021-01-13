import numpy as np
import pandas as pd
import random  
import os.path 

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import units
from pysc2.lib import features

DATA_FILE = 'sparse_agent_Data'


ACTION_ATTACK = 'attack'
ACTION_RETREAT = 'retreat'

smartactions = [
        ACTION_ATTACK,
        ACTION_RETREAT    
    ]

def _xy_locs(mask):
    y, x = np.nonzero(mask)
    return list(zip(x, y))


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):

    
 # Gets all the players units by unit type
    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    # Gets all the enemy units by type 
    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]    

    def splitAction(self, action_id):
        smart_action = smartactions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)


    def __init__(self):
        super(SparseAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smartactions))))

        self.previous_action = None
        self.previous_state = None

        # Saves the q learn table to a file if it doesnt already
        # If it is load the data into the algorithm
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def attack(self, obs):

        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            targets = _xy_locs(player_relative == features.PlayerRelative.ENEMY)

            if not targets:
                return actions.FUNCTIONS.no_op()

            target = targets[np.argmax(np.array(targets)[:, 1])]
            return actions.FUNCTIONS.Attack_screen("now", target)

        if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army("select")
        
        return actions.FUNCTIONS.no_op()

    def retreat(self, obs):

        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:

            player_relative = obs.observation.feature_screen.player_relative
            enemies = _xy_locs(player_relative == features.PlayerRelative.ENEMY)

            if not enemies:
                return actions.FUNCTIONS.no_op()
            
            distance = 5


            myUnits = [unit for unit in obs.observation.feature_units
                 if unit.alliance == features.PlayerRelative.SELF]
            if(myUnits[0].x - distance <= 0 or myUnits[0].y - distance <= 0):
                return actions.FUNCTIONS.no_op()
            else:
                moveAway = (myUnits[0].x - distance, myUnits[0].y - distance)         


            return actions.FUNCTIONS.Move_screen("now", moveAway)


        return actions.FUNCTIONS.select_army("select")

    def step(self, obs):
        super(SparseAgent, self).step(obs)

        if obs.last():
            reward = obs.reward 

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0 

       # if obs.first():
        
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        eneimes = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)

        # Loops through all the units that can be seen and if they're friendly add them to the array 
        myUnits = [unit for unit in obs.observation.feature_units
                 if unit.alliance == features.PlayerRelative.SELF]

        enemyUnits = [unit for unit in obs.observation.feature_units
                    if unit.alliance == features.PlayerRelative.ENEMY]

        marine_health = 0
        enemy_health = 0
        # Check to see if there is a player alive 
        if(len(myUnits) > 0 and len(enemyUnits) > 0):
            marine_health = myUnits[0].health
            enemy_health = enemyUnits[0].health


        current_state = np.zeros(8)
       # current_state[0] = marines
       # current_state[1] = eneimes
       # current_state[2] = marine_health
       # current_state[3] = enemy_health
        
        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))

        self.previous_state = current_state
        self.previous_action = rl_action

        smart_action, x, y = self.splitAction(self.previous_action)

        if smart_action == ACTION_ATTACK:
            self.attack(obs)
        
        elif smart_action == ACTION_RETREAT:
            self.retreat(obs)




        


