import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

class StrafBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_gateway()
        await self.build_assimilator()
        await self.build_cyber_core()
        await self.train_stalkers()
        await self.build_four_gates()

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
        pos = nexus.position.towards(self.enemy_start_locations[0], 10)

        if(
            self.supply_left < 3 
            and self.already_pending(UnitTypeId.PYLON) == 0
            and self.can_afford(UnitTypeId.PYLON)
        ):
            await self.build(UnitTypeId.PYLON, near = pos)

    async def build_gateway(self):
        if(
            self.structures(UnitTypeId.PYLON).ready
            and self.can_afford(UnitTypeId.GATEWAY)
            and not self.structures(UnitTypeId.GATEWAY)
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near = pylon)

    async def build_assimilator(self):
        if self.structures(UnitTypeId.GATEWAY):
            for nexus in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(15, nexus)
                for vg in vgs: 
                    if not self.can_afford(UnitTypeId.ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vg.position)
                    if worker is None:
                        break
                    if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                        worker.build(UnitTypeId.ASSIMILATOR, vg)
                        worker.stop(queue = True)
    
    async def build_cyber_core(self):
        if self.structures(UnitTypeId.PYLON).ready:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if self.structures(UnitTypeId.GATEWAY).ready:
                #if no cyber core build one
                if not self.structures(UnitTypeId.CYBERNETICSCORE):
                    if(
                        self.can_afford(UnitTypeId.CYBERNETICSCORE)
                        and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                    ):
                        await self.build(UnitTypeId.CYBERNETICSCORE, near = pylon)
            
    async def train_stalkers(self):
        for gateway in self.structures(UnitTypeId.GATEWAY).ready:
            if(
                self.can_afford(UnitTypeId.STALKER)
                and gateway.is_idle
            ):
                gateway.train(UnitTypeId.STALKER)

    async def build_four_gates(self):
        if(
            self.structures(UnitTypeId.PYLON).ready
            and self.can_afford(UnitTypeId.GATEWAY)
            and self.structures(UnitTypeId.GATEWAY).amount < 4
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near = pylon)





run_game(maps.get("AbyssalReefLE"),[
    Bot(Race.Protoss, StrafBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime = True)