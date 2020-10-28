import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

class StrafBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()

    async def build_workers(self):
        #Gets the nexus in game 
        nexus = self.townhalls.ready.random

        #Checks to see if we can afford to build the worker
        #And if the nexus is idle and we're not over the worker cap 
        if(
            self.can_afford(UnitTypeId.PROBE)
            and nexus.is_idle
            and self.workers.amount < self.townhalls.amount * 22
        ):
        #Train the unit 
            nexus.train(UnitTypeId.PROBE)

    async def build_pylons(self):
        #Gets the nexus in game 
        nexus = self.townhalls.ready.random
        



run_game(maps.get("AbyssalReefLE"),[
    Bot(Race.Protoss, StrafBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime = True)