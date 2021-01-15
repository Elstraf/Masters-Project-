import numpy as np 
import pandas as pd
import random 
import os.path
import math

from pysc2.agents import base_agent
from pysc2.lib import actions 
from pysc2.lib import units
from pysc2.lib import features


## TO RUN THE AGENT:   python -m pysc2.bin.agent --map DefeatWhatever --agent pysc2.agents.QLearningMiniGame.Agent --use_raw_units --use_feature_units
# python -m pysc2.bin.agent --map DefeatWhatever --agent pysc2.agents.QLearningMiniGame.Agent --use_raw_units --use_feature_units --max_episodes 1000

DATA_FILE = 'qlearning_agent_data'


def _xy_locs(mask):
    y, x = np.nonzero(mask)
    return list(zip(x, y))


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))



class Agent(base_agent.BaseAgent):

    smartActions = [
        "retreat",
        "attack"
    ]
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

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)               
                         

    def __init__(self):
        super().__init__()

        self.qtable = QLearningTable(self.smartActions)

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

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
            
            distance = 2

            myUnits = [unit for unit in obs.observation.feature_units
                if unit.alliance == features.PlayerRelative.SELF]
            enemyUnits = [unit for unit in obs.observation.feature_units
                if unit.alliance == features.PlayerRelative.ENEMY]

            distanceBetween = math.hypot(myUnits[0].x - enemyUnits[0].x, myUnits[0].y - enemyUnits[0].y)

            if(myUnits[0].x - distance <= 0 or myUnits[0].y - distance <= 0):
                return actions.FUNCTIONS.no_op()

            moveAway = (myUnits[0].x - distance, myUnits[0].y - distance)
                
                #if(myUnits[0].x - distance <= 0 or myUnits[0].y - distance <= 0):
                    #return actions.FUNCTIONS.no_op()
                #else:
                    #moveAway = (myUnits[0].x - distance, myUnits[0].y - distance)         


            return actions.FUNCTIONS.Move_screen("now", moveAway)


        return actions.FUNCTIONS.select_army("select")

    def get_state(self, obs):

        #Get all the agents and enemies units 
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        eneimes = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)

        # Loops through all the units that can be seen and if they're friendly add them to the array 
        myUnits = [unit for unit in obs.observation.feature_units
                 if unit.alliance == features.PlayerRelative.SELF]

        # Check to see if there is a player alive 
        if(len(myUnits) > 0):
            marine_health = myUnits[0].health
            #return the hp of the agents unit
            return((len(marines),
                len(eneimes),
                marine_health))

        enemyUnits = [unit for unit in obs.observation.feature_units
                    if unit.alliance == features.PlayerRelative.ENEMY]

        # Check to see if there is a player alive 
        if(len(myUnits) > 0 and len(enemyUnits) > 0):
            marine_health = myUnits[0].health
            enemy_health = enemyUnits[0].health
            #return the hp of the agents unit
            return((len(marines),
                len(eneimes),
                marine_health,
                enemy_health
                ))

        return((len(marines),
                len(eneimes),
               ))

    def reset(self):
        super(Agent, self).reset()
        self.new_game()

    def new_game(self):
        self.previous_state = None
        self.previous_action = None


    def step(self, obs):
        super(Agent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)


        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward,
                              'terminal' if obs.last() else state)
            
        self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)


        