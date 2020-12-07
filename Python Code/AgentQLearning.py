#Imports 
import numpy as np 
import pandas as pd
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

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


#Gets all enemies on the screen 
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

#Gets all player units on screen 
_PLAYER_SELF = features.PlayerRelative.SELF


class QLearningAgent(base_agent.BaseAgent):
    #Agent that will beat the minigame with Q Learning

    #Actions that the AI will be able to us - Simple list at first while I get it working
    actions = ("do_nothing",
               "attack",
               "retreat")

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    #Gets the distance between self and objects
    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            roaches =0 #[units for unit in obs.observation.raw_units
                       #if unit.alliance == _PLAYER_ENEMY]
            distances = self.get_distances(obs, marines, roaches)
            marine = marines[np.argmax(distances)]

            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (roaches[0])
            )
        return actions.RAW_FUNCTIONS.no_op()

    def retreat(self, obs):
        #Returns nothing for now edit this later 
        return actions.RAW_FUNCTIONS.no_op()

    def get_state(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        enemyRoaches =0#[units for unit in obs.observation.raw_units
                        #if unit.alliance == _PLAYER_ENEMY]
        
        return(len(marines),
               len(enemyRoaches))

    def __init__(self):
        super(QLearningAgent, self).__init__()
        self.qtable = QLearningTable(self.actions)

    def new_game(self):
        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        super(QLearningAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward,
                              'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)