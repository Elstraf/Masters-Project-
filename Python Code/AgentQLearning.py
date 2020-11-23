#Imports 
import numpy 

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from QLearning import QLearningTable 


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

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)



    def step(self, obs):
        super(QLearningAgent, self).step(obs)

        return actions.RAW_FUNCTIONS.no_op()
        