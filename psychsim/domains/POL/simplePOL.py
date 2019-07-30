from psychsim.reward import *
from psychsim.pwl import *
from psychsim.action import *
from psychsim.world import *
from psychsim.agent import *
import random
from random import randint
from time import time

class Person(Agent):
	def __init__(self, name, init_location, world, feelings_toward, random_horizon=False, random_weights=False, random_costs=False):
		Agent.__init__(self,name)
		world.addAgent(self)

		# set the horizon to 2, so he can see that it should make money to eat
		
		horizon_value=randint(1,4) if random_horizon else 2
		print('\n'+name+':')
		print(' Horizon: '+str(horizon_value)+'\n')
		self.setAttribute('horizon',horizon_value)

		
		# define features of a person
		current_x=world.defineState(self.name,'current_x')
		current_y=world.defineState(self.name,'current_y')
		wealth=world.defineState(self.name,'wealth')
		hunger=world.defineState(self.name,'hunger')
		comfort=world.defineState(self.name,'comfort')
		health=world.defineState(self.name,'health')


		# initialize the features
		self.setState('current_x', init_location[0])
		self.setState('current_y', init_location[1])
		self.setState('wealth', 0.)
		self.setState('hunger', 0.)
		self.setState('comfort', 0.)
		self.setState('health', 100.)


		# define rewards
		wealth_weight  =randint(3,6)*1. if random_weights else 3.
		comfort_weight =randint(3,6)*1. if random_weights else 4.
		health_weight  =randint(3,6)*1. if random_weights else 5.
		hunger_weight  =randint(3,6)*1. if random_weights else 6.
		print(' Wealth Weight: ' +str(wealth_weight ))
		print(' Comfort Weight: '+str(comfort_weight))
		print(' Health Weight: ' +str(health_weight ))
		print(' Hunger Weight: ' +str(hunger_weight )+'\n')
		self.setReward(maximizeFeature(wealth,self.name),  wealth_weight)## maximize
		self.setReward(maximizeFeature(comfort,self.name),comfort_weight)
		self.setReward(maximizeFeature(health,self.name),  health_weight)
		self.setReward(minimizeFeature(hunger,self.name),  hunger_weight)## minimize


		# define actions
		time2work=8
		time2home=16
		time_pass = stateKey('current_environment','time_pass') # point to the same variable
		tree= makeTree({'if': thresholdRow(time_pass,time2work), # if time_pass is larger than time2work, larger than 8
						True: {'if': thresholdRow(time_pass,time2home), # if the time_pass is less than time2home, less than 16
								True: False,
								False: True},
						False: False})
		work = self.addAction({'verb':'work'}, tree)

		
		eating_cost =randint(10,50)*.01 if random_costs else .5
		wealth_cost =randint(10,20)*.01 if random_costs else .1 
		hunger_cost =randint(10,20)*.01 if random_costs else .1
		comfort_cost=randint(10,20)*.01 if random_costs else .1
		print(' Eating Cost: ' +str(eating_cost ))
		print(' Wealth Cost: ' +str(wealth_cost ))
		print(' Hunger Cost: ' +str(hunger_cost ))
		print(' Comfort Cost: '+str(comfort_cost))
		print('------------------------------------')


		# see if wealth is larger than the eating cost before eating
		tree= makeTree({'if': thresholdRow(wealth,eating_cost),
						True: True,
						False: False})
		eat = self.addAction({'verb':'eat'}, tree)
		gohome = self.addAction({'verb':'gohome'})

		# set dynamics
		commercial=world.agents['commercial']
		comm_width_range=(commercial.center_x-(commercial.width/2), commercial.center_x+(commercial.width/2))
		comm_length_range=(commercial.center_y-(commercial.length/2), commercial.center_y+(commercial.length/2))
		
		residential=world.agents['residential']
		resi_width_range=(residential.center_x-(residential.width/2), residential.center_x+(residential.width/2))
		resi_length_range=(residential.center_y-(residential.length/2), residential.center_y+(residential.length/2))
		
		## impacts of work
		tree = makeTree(setToConstantMatrix(current_x,randint(comm_width_range[0],comm_width_range[1])))
		world.setDynamics(current_x,work,tree)
		tree = makeTree(setToConstantMatrix(current_y,randint(comm_length_range[0],comm_length_range[1])))
		world.setDynamics(current_y,work,tree)
		### instead of increments by a constant, we increase by a percentage, so that the changes in one need will be proportionally reflected in the multiplications. That is, instead of weights being multiplied by a certian value each time, they will be multiplied by the amount of change. This way, if one need is not satified, over time, it will get more attention.
		tree = makeTree(approachMatrix(wealth,wealth_cost,100))
		world.setDynamics(wealth,work,tree)

		tree = makeTree(approachMatrix(hunger,hunger_cost,100))
		world.setDynamics(hunger,work,tree)

		tree = makeTree(approachMatrix(comfort,comfort_cost,100))
		world.setDynamics(comfort,work,tree)

		risk=stateKey(commercial,'risk')
		tree = makeTree(approachMatrix(health,risk,0))
		world.setDynamics(comfort,work,tree)

		## impacts of eat
		tree = makeTree(approachMatrix(hunger,.9,0))
		world.setDynamics(hunger,eat,tree)

		tree = makeTree(approachMatrix(wealth,eating_cost,0))
		world.setDynamics(wealth,eat,tree)

		## impacts of gohome
		tree = makeTree(setToConstantMatrix(current_x,randint(resi_width_range[0],resi_width_range[1])))
		world.setDynamics(current_x,gohome,tree)
		tree = makeTree(setToConstantMatrix(current_y,randint(resi_length_range[0],resi_length_range[1])))
		world.setDynamics(current_y,gohome,tree)

		tree = makeTree(approachMatrix(comfort,comfort_cost,100))
		world.setDynamics(comfort,gohome,tree)

		tree = makeTree(approachMatrix(hunger,hunger_cost,100))
		world.setDynamics(hunger,gohome,tree)

	def init_relations(self, world, external_force, relation_value=0.): #relation is a float in the range of [-1.,1.]
		relation= self.world.defineRelation(self.name,external_force,'feelings_toward',float)
		self.world.setFeature(relation,0.)

		# here we can access another object which in this case is the external_force
		external_force_obj=world.agents[external_force]
		
		tree = makeTree(approachMatrix(relation,.1,1.))
		world.setDynamics(relation,external_force_obj.do_good,tree)
		tree = makeTree(approachMatrix(relation,.1,-1.))
		world.setDynamics(relation,external_force_obj.do_bad,tree)


class Region(Agent):
	def __init__(self, name, dimentions, center, max_occupancy, world):
		Agent.__init__(self,name)
		world.addAgent(self)
		self.width, self.length = dimentions
		self.center_x, self.center_y= center
		self.max_occupancy=max_occupancy

		current_occupancy = world.defineState(self.name,'current_occupancy')
		self.setState('current_occupancy', 0)

		#risk/danger level of being in this region for a civilian health
		risk = world.defineState(self.name,'risk')
		self.setState('risk', 0)

class Environment(Agent):
	def __init__(self, name, world):
		Agent.__init__(self,name)
		world.addAgent(self)
		time_pass=world.defineState(self.name,'time_pass')

		time_increase = self.addAction({'verb':'time_increase'})
		self.setState('time_pass', 0)

		# impact of time_increase
		tree = makeTree(incrementMatrix(time_pass,1))
		world.setDynamics(time_pass,time_increase,tree)

class ArmedForces(Agent):
	def __init__(self, name, world, init_location, size):
		Agent.__init__(self,name)
		world.addAgent(self)

		self.setState('current_x', init_location[0])
		self.setState('current_y', init_location[1])

		do_good_cost=-10
		do_bad_cost=10
		regions = [name for name in self.world.agents if isinstance(self.world.agents[name],Region)]
		# here we have to write it declaratively because it is not executed now. so we define the actions over all of the regions and their risks but here in the Armed_Forced class where the actions impact the risks
		for region in regions:
			do_good = self.addAction({'verb':'do_good'})
			do_bad = self.addAction({'verb':'do_bad'})
			risk=stateKey(region,'risk')
			tree = makeTree(approachMatrix(risk,do_good_cost,0))
			world.setDynamics(risk,do_good,tree)
			tree = makeTree(approachMatrix(risk,do_bad_cost,100))
			world.setDynamics(risk,do_bad,tree)

		
world = World()
residential = Region('residential', (20,20), (40,40), 100, world)

commercial = Region('commercial', (10,10), (80,40), 25, world)


environment_agent =Environment('current_environment', world)

# civilian1 = Person('civilian1', (25, 35), world)

resi_width_range=(residential.center_x-(residential.width/2), residential.center_x+(residential.width/2))
resi_length_range=(residential.center_y-(residential.length/2), residential.center_y+(residential.length/2))

start_time=time()
civilians=[]
for j in range(10):
	starting_location=(randint(resi_width_range[0], resi_width_range[1]), randint(resi_length_range[0], resi_length_range[1]))
	civilian = Person('civilian %d' %(j+1), starting_location, world, True, True, True)
	civilians.append(civilian.name)

# world.setOrder([{'civilian1'}])
world.setOrder([set(civilians) | {environment_agent.name}])

for i in range(24):
	start_round=time()
	print('\n \tStep %d: ' %(i+1))
	newState = world.step()
	#world.explainAction(newState)
	world.printState(newState)
	end_round=time()
	print('Step time: '+str(round(end_round-start_round)))
	print('----------------------------------------------------------------------')

end_time=time()
print('Overall time; '+str(round(end_time-start_time)))

# print('x: ' + str(world.getState(civilian1.name,'x')) + ' , y: ' + str(world.getState(civilian1.name,'y')))
# print('goal x: ' + str(world.getState(civilian1.name,'goal_x')) + ' , goal y: ' + str(world.getState(civilian1.name,'goal_y')))
