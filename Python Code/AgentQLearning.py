#Imports 
import numpy as np 

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from QLearning import QLearningTable 


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

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            roaches = [units for unit in obs.observation.raw_units
                       if unit.alliance == _PLAYER_ENEMY]
            distances = self.get_distances(obs, marines, roaches)
            marine = marines[np.argmax(distances)]

            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (roaches[0])
            )
        return actions.RAW_FUNCTIONS.no_op()


    def step(self, obs):
        super(QLearningAgent, self).step(obs)

        return actions.RAW_FUNCTIONS.no_op()
        