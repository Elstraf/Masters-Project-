import numpy as np 
import pandas as pd
import random 

from pysc2.agents import base_agent
from pysc2.lib import actions 
from pysc2.lib import units
from pysc2.lib import features


## TO RUN THE AGENT:   python -m pysc2.bin.agent --map DefeatWhatever --agent pysc2.agents.QLearningMiniGame.Agent

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

        self.qlearn = QLearningTable(self.smartActions)

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
            
            moveAway = (random.randint(0, 100), random.randint(0,100))         

            return actions.FUNCTIONS.Move_screen("now", moveAway)


        return actions.FUNCTIONS.select_army("select")



    def step(self, obs):
        super(Agent, self).step(obs)

        # Gets all the players units on screen
        # Since its just one set of units we dont need to check for certain types 
        #marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        # Same as above just for the enemy ones
        #targets = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)

       # marines = [unit.tag for unit in obs.observation.raw_units
            #    if unit.alliance == features.PlayerRelative.SELF]
     #   targets = [unit for unit in obs.observation.raw_units
              #  if unit.alliance == features.PlayerRelative.ENEMY]
        
        #return actions.RAW_FUNCTIONS.no_op()

                # select the user units 


        