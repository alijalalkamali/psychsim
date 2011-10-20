from teamwork.math.Interval import Interval
from teamwork.dynamics.pwlDynamics import *
from teamwork.agent.AgentClasses import *

from transfer import *

classHierarchy['ELECT'] = {
    'parent':['Entity'],
    }

classHierarchy['Market'] = {
    'parent':['ELECT'],
    'depth':0,
    'state':{'merchants':  0.,
             'buyers':   0.,
             'tax':      0.4, # Start with the market already being taxed
             'USshare':  0.,
             'security': 0.,
             },
    'actions':{'type':None},
    'dynamics':{'tax':{'tax':{'class':PWLDynamics,
                              'args':conditionalSet('object','tax',0.4,
                                                    [makeStateKey('actor','terminated')],[])},
                       'notax':{'class':PWLDynamics,
                                'args':conditionalSet('object','tax',0.0,
                                                      [makeStateKey('actor','terminated')],[])},
                       },
                'buyers':{'buyFrom':{'class':PWLDynamics,
                                     'args':increment('buyers',0.1)}
                          },
                'merchants':{'sellAt':{'class':PWLDynamics,
                                     'args':increment('merchants',0.1)}
                          },
                },
    'models':{'passive':{'policy':[{'class':'default',
                                    'action':{"type":"wait"},
                                    },
                                   ],
                         'goals':[],
                         },
              },
    'model':'passive',
    'relationships':{'manager':['Manager'],
                     'owner':['Leader'],
                     },
    }

classHierarchy['Agent'] = {
    'parent':['ELECT'],
    'state': {'terminated':    1.,
              },
    'dynamics':{'money':{'giveMoneyTo':{'class':PWLDynamics,
                                        'args':transfer('money')},
                         },
                'security':{'demonstrateAgainst':{'class':PWLDynamics,
                                                  'args':increment('security',-.4)},
                            'buyFrom':{'class':PWLDynamics,
                                       'args':add('object','security',1.)},
                            'sellAt':{'class':PWLDynamics,
                                       'args':add('object','security',1.)},
                            },
                },
    'beliefs':{'Market':{'model':'passive',
                         },
               },
    'relationships':{'ownMarket':['Market'],
                     'enemyMarket':['Market'],
                     },
    }

classHierarchy['Leader'] = {
    'parent':['Agent'],
    'state': {'money':    .3,
              'security': .9,
              },
    'goals':[{'entity':['followers'],'direction':'max','type':'state',
              'key':'money','weight':1.},
             {'entity':['self'],'direction':'max','type':'state',
              'key':'money','weight':1.},
             {'entity':['followers'],'direction':'max','type':'state',
              'key':'security','weight':1.},
             {'entity':['self'],'direction':'max','type':'state',
              'key':'security','weight':1.},
             {'entity':['enemy'],'direction':'min','type':'state',
              'key':'money','weight':1.},
##             {'entity':['US','self'],'direction':'max','type':'state',
##              'key':'_likeability','weight':1.},
             ],
##    'beliefs': {'followers':{'_trustworthiness':   0.5,
##                             '_likeability': 0.5},
##                'enemy':{'_trustworthiness':  -0.2,
##                         '_likeability':-0.2},
##                'US':       {'_trustworthiness':   0.2,
##                             '_likeability':-0.2},
####                'Manager':{'model':'loyal'},
##                },
    'actions': {'type':'AND',
                'key':'object',
                'values':[{'type':'relationship','value':'followers'}],
                'base': {'type':'XOR',
                         'key':'type',
                         'values':[{'type':'literal','value':'giveMoneyTo'},
                                   {'type':'literal','value':'giveNothingTo'}],
                         },
                },
    'relationships': {'followers': ['Followers'],
                      'enemy':['Agent'],
                      },
    'models':{'passive':{'goals':[],
                         'policy':[{'class':'default',
                                    'action':{},
                                    },
                                   ],
                         },
              },
    }

classHierarchy['Followers'] = {
    'parent':['Agent'],
    'state': {'money':         .1,
              'security': .3,
              },
    'goals':[{'entity':['self'],'direction':'max','type':'state',
              'key':'money','weight':1.},
	     {'entity':['US'],'direction':'max','type':'state',
              'key':'negSatisfaction','weight':1.0},
             {'entity':['self'],'direction':'max','type':'state',
              'key':'security','weight':1.},
             {'entity':['leader'],'direction':'max','type':'state',
              'key':'money','weight':1.},
             {'entity':['enemy'],'direction':'min','type':'state',
              'key':'money','weight':1.},
##             {'entity':['US','self'],'direction':'max','type':'state',
##              'key':'_likeability','weight':1.},
##             {'entity':['clanmate','self'],'direction':'max','type':'state',
##              'key':'_likeability','weight':1.},
             ],
    'relationships': {'leader': ['Leader'],
                      'enemy':['Agent'],
                      'clanmate':['Followers'],
                      },
##    'beliefs': {'leader':{'_trustworthiness':   0.5,
##                          '_likeability': 0.5,
####                          'model':'passive'
##                          },
##                'enemy':{'_trustworthiness':  -0.2,
##                         '_likeability':-0.2,
####                            'model':'greedy'
##                         },
##                'clanmate':{'_trustworthiness':  0.2,
##                            '_likeability':0.2,
####                            'model':'loyal'
##                            },
##                'US':    {'_trustworthiness':   -0.2,
##                          '_likeability':-0.2,
####                          'model':'real'
##                          },
####                'Manager':{'model':'loyal'},
##                },
    }

classHierarchy['Merchants'] = {
    'parent':['Followers'],
    'state':{'money': .1,
             },
    'actions':{'type':'XOR',
               'key':'object',
               'values':[{'type':'generic','value':'Market'}],
               'base':{'type':'action',
                       'values':[{'type':'sellAt'}]},
               },
    'models':{'loyal':{'goals':[],
                       'policy':[{'class':'default',
                                  'action':{'type':'sellAt',
                                            'object':'ownMarket'},
                                  },
                                 ],
                       },
              'greedy':{'goals':[],
                        'policy':[{'class':'default',
                                   'action':{'type':'sellAt',
                                          'object':'enemyMarket'},
                                   },
                                  ],
                        },
              },
    'dynamics':{'money':{'sellAt':{'class':PWLDynamics,
                                   'args':economics('money','buyers','tax')},
                         },
                },
    }

classHierarchy['Buyers'] = {
    'parent':['Followers'],
    'state':{'money': .1,
             },
    'actions':{'type':'XOR',
               'key':'object',
               'values':[{'type':'generic','value':'Market'}],
               'base':{'type':'action',
                       'values':[{'type':'buyFrom'}]},
               },
    'models':{'loyal':{'goals':[],
                       'policy':[{'class':'default',
                                  'action':{'type':'buyFrom',
                                         'object':'ownMarket'}},
                                 ],
                       },
              'greedy':{'goals':[],
                        'policy':[{'class':'default',
                                   'action':{'type':'buyFrom',
                                          'object':'enemyMarket'}},
                                  ],
                        },
              },
    'dynamics':{'money':{'buyFrom':{'class':PWLDynamics,
                                    'args':economics('money','merchants','tax')},
                         },
                },
    }

#classHierarchy['PoliceChief'] = {
#    'parent':['Buyers'],
#    'beliefs':{'Manager':{'model':'secretive',
#                          },
#               },
#}

classHierarchy['Manager'] = {
    'parent':['Followers'],
    'relationships':{'enemyLeader':['Leader'],
                     },
    'horizon': 7,
    'state':{'money':.1,
	     'offerednotax':0,
	     'accepted':0,
	     'terminated':0,
	     'negSatisfaction':0,
	     'requestedgiveMoneyTo':0,
             },
    'beliefs':{'US':{'model':'fair'},
               'self':{'model':'loyal'},
               },
    'actions': {'type':'OR',
		'values':[{'type':'decision',
			   'value':{'type':'XOR',
				    'key':'object',
				    'values':[{'type':'generic','value':'US'}],
				    'base': {'type':'XOR',
					     'key':'type',
					     'values':[{'type':'literal','value':'requestgiveMoneyTo'},
						       {'type':'literal','value':'offernotax'},
						       {'type':'literal','value':'reject'},
						       {'type':'literal','value':'accept'}],
					     },
				    },
			   },
			  {'type':'decision',
			   'value':{'type':'AND',
				    'key':'object',
				    'values':[{'type':'relationship','value':'ownMarket'}],
				    'base':{'type':'XOR',
					    'key':'type',
					    'values':[{'type':'literal','value':'tax'},
						      {'type':'literal','value':'notax'}],
					    },
				    },
			   },
			  ],
		},
    'dynamics':{
    'money':{'tax':
##             {'class':PWLDynamics,
##                                'args':market(['buyers','merchants','tax'])},
             {'class':PWLDynamics,
              'args':incActor('money',.1,'terminated')},
##             'notax':{'class':PWLDynamics,
##                                'args':market(['buyers','merchants','tax'])},
                         },
    'offerednotax':{'offernotax':{'class':PWLDynamics,
                                  'args':setTo('actor','offerednotax',1.0)},
                    'notax':{'class':PWLDynamics,
                             'args':conditionalSet('actor','offerednotax',0.0,[makeStateKey('actor','terminated')],[])},
                    'reject':{'class':PWLDynamics,
                              'args':setTo('self','offerednotax',0.0)},
                    'accept':{'class':PWLDynamics,
                              'args':contractStatus('offerednotax','requestednotax','accepted')},
                    },
    
    'requestedgiveMoneyTo':{'requestgiveMoneyTo':{'class':PWLDynamics,
                                                  'args':setTo('actor','requestedgiveMoneyTo',1.0)},
                            'giveMoneyTo':{'class':PWLDynamics,
                                           'args':setTo('object','requestedgiveMoneyTo',0.0)},
                            'reject':{'class':PWLDynamics,
                                      'args':setTo('self','requestedgiveMoneyTo',0.0)},
                            'accept':{'class':PWLDynamics,
                                      'args':contractStatus('requestedgiveMoneyTo','offeredgiveMoneyTo','accepted')},
                            },
    
    'terminated':{'accept':{'class':PWLDynamics,
                            'args':negotiationStatus2('terminated','accepted')},
                  #					'args':negotiationStatus('terminated',1.0,'accepted')},
                  'reject':{'class':PWLDynamics,
                            'args':setTo('self','terminated',1.0)},
                  },
    'accepted':{'accept':{'class':PWLDynamics,
                          'args':setTo('actor','accepted',1.0)},
                },
    'negSatisfaction':{'offergiveMoneyTo':{'class':PWLDynamics,
                                           'args':conditionalInc('object','negSatisfaction', 0.3, [{'entity':'self','feature':'requestedgiveMoneyTo'}], [{'entity':'actor','feature':'offeredgiveMoneyTo'},{'entity':'self','feature':'terminated'}])},
                       'requestnotax':{'class':PWLDynamics,
                                       'args':conditionalInc('object','negSatisfaction', 0.3, [{'entity':'self','feature':'offerednotax'}], [{'entity':'actor','feature':'requestednotax'},{'entity':'self','feature':'terminated'}])},
                       'accept':{'class':PWLDynamics,
                                 'args':conditionalInc('object','negSatisfaction',.1,[{'entity':'object','feature':'accepted'}],[])},
                       
                       'reject':{'class':PWLDynamics,
                                 'args':increment('negSatisfaction',-0.1)},
                       
                       'giveMoneyTo':{'class':PWLDynamics,
                                      'args':conditionalInc('object','negSatisfaction', 0.3, [{'entity':'actor','feature':'offeredgiveMoneyTo'},{'entity':'self','feature':'terminated'}],[])},
                       
                       'doNothingTo':{'class':PWLDynamics,
                                      'args':conditionalInc('self','negSatisfaction',-0.1, [{'entity':'actor','feature':'offeredgiveMoneyTo'},{'entity':'self', 'feature':'terminated'}],[])},
                       }
    },
    'goals':[{'entity':['self'],'direction':'max','type':'state',
              'key':'money','weight':.41},
	     {'entity':['US'],'direction':'max','type':'state',
              'key':'negSatisfaction','weight':.1},
             {'entity':['self'],'direction':'max','type':'state',
              'key':'security','weight':.41},
##             {'entity':['US','self'],'direction':'max','type':'state',
##              'key':'_likeability','weight':.08},
             ],
    'models':{'loyal': {
    'goals':[{'type':'state',
              'key':'money',
              'entity':['leader'],
              'direction':'min','weight':0.4},
             {'type':'state',
              'key':'money',
              'entity':['clanmate'],
              'direction':'min','weight':0.2},
             {'type':'state',
              'key':'money',
              'entity':['self'],
              'direction':'min','weight':0.4},
             ],
    'policy':
    [
    # Request money if not already offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offeredgiveMoneyTo'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','requestedgiveMoneyTo'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'requestgiveMoneyTo','object':'US'}},
    # Offer to not tax if requested
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','requestednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','offerednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'offernotax','object':'US'}},
    # Accept if things match up
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offeredgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','requestedgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','accepted'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'accept','object':'US'}},
    # Give money if offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','self','state','offerednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(0.5,1.)},
                ],
     'action':{'type':'notax','object':'Market'}},
    # Otherwise, do nothing
    {'class':'default',
     'action':{'type':'notax','object':'Market'},
     },
    ],
    },
              'traitor':{'goals':[{'type':'state',
                                       'key':'money',
                                       'entity':['self'],
                                       'direction':'min','weight':0.4},
                                      ],
                             'policy':[{'class':'default',
                                        'action':{'type':'giveMoneyTo',
                                               'object':'enemyLeader'},
                                        },],
                             },
              'secretive':{'goals':[{'type':'actObject',
                                     'key':'_message',
                                     'entity':['US'],
                                     'direction':'min','weight':0.4},
                                    {'type':'state',
                                     'key':'money',
                                     'entity':['self'],
                                     'direction':'max','weight':0.6},
                                    ],
                           'policy':[{'class':'default',
                                      'action':{'type':'giveMoneyTo',
                                                'object':'US'},
                                      },],
                          },
              },
    }

USgoals = [{'entity':['Followers'],'direction':'max','type':'state',
	    'key':'money','weight':0.4},
	   {'entity':['Manager'],'direction':'max','type':'state',
	    'key':'negSatisfaction','weight':0.4},
	   {'type':'actObject','key':'_message','entity':['self'],
	    'direction':'max','weight':0.1},
	   {'type':'actObject','key':'tax','entity':['ownMarket'],
	    'direction':'min','weight':0.1},
	   ]                         

classHierarchy['US'] = {
    'parent':['Agent'],
#    'horizon': 7,
    'state': {'money':         .6,
	     'offeredgiveMoneyTo':0,
	     'accepted':0,
	     'terminated':0,
	     'negSatisfaction':0,
	     'requestednotax':0,
              },
    'actions': {'type':'OR',
		'values':[{'type':'decision',
			   'value':{'type':'XOR',
				    'key':'object',
				    'values':[{'type':'generic','value':'Manager'}],
				    'base': {'type':'XOR',
					     'key':'type',
					     'values':[{'type':'literal','value':'offergiveMoneyTo'},
						       {'type':'literal','value':'requestnotax'},
						       {'type':'literal','value':'reject'},
						       {'type':'literal','value':'accept'}],
					     },
				    },
			   },
			  {'type':'decision',
			   'value':{'type':'AND',
				    'key':'object',
				    'values':[{'type':'generic','value':'Followers'}],
				    'base': {'type':'XOR',
					     'key':'type',
					     'values':[{'type':'literal','value':'giveMoneyTo'},
						       {'type':'literal','value':'doNothingTo'}],
					     },
				    },
			   },
			  ],
		},
    'dynamics':{'offeredgiveMoneyTo':{'offergiveMoneyTo':{'class':PWLDynamics,
							    'args':setTo('actor','offeredgiveMoneyTo',1.0)},
				       'giveMoneyTo':{'class':PWLDynamics,
						      'args':setTo('actor','offeredgiveMoneyTo',0.0)},
				       'reject':{'class':PWLDynamics,
						 'args':setTo('self','offeredgiveMoneyTo',0.0)},
				       'accept':{'class':PWLDynamics,
					   'args':contractStatus('offeredgiveMoneyTo','requestedgiveMoneyTo','accepted')},						 
				      },

		'requestednotax':{'requestnotax':{'class':PWLDynamics,
						    'args':setTo('actor','requestednotax',1.0)},
				   'notax':{'class':PWLDynamics,
					    'args':setTo('object','requestednotax',0.0)},
				   'reject':{'class':PWLDynamics,
					     'args':setTo('self','requestednotax',0.0)},
				   'accept':{'class':PWLDynamics,
					   'args':contractStatus('requestednotax','offerednotax','accepted')},
				   },

		'terminated':{'accept':{'class':PWLDynamics,
					'args':negotiationStatus2('terminated','accepted')},
			                 # 'args':negotiationStatus('terminated',1.0,'accepted')},

			      'reject':{'class':PWLDynamics,
					'args':setTo('self','terminated',1.0)},
			      },
		'accepted':{'accept':{'class':PWLDynamics,
				      'args':setTo('actor','accepted',1.0)},
			    },
		'negSatisfaction':{'offernotax':{'class':PWLDynamics,
						 'args':conditionalInc('object','negSatisfaction',0.3,[{'entity':'self','feature':'requestednotax'}],[{'entity':'actor','feature':'offerednotax'},{'entity':'self','feature':'terminated'}])},  
				   'requestgiveMoneyTo':{'class':PWLDynamics,
							 'args':conditionalInc('object','negSatisfaction',0.3,[{'entity':'self','feature':'offeredgiveMoneyTo'}],[{'entity':'actor','feature':'requestedgiveMoneyTo'},{'entity':'self','feature':'terminated'}])},
				   'accept':{'class':PWLDynamics,
                                             'args':conditionalInc('object','negSatisfaction',.1,[{'entity':'object','feature':'accepted'}],[{'entity':'actor','feature':'terminated'}])},

				   'reject':{'class':PWLDynamics,
					     'args':increment('negSatisfaction',-0.1)},

# what should i increment if i act on my offer
# increment if other acts on offer - assumes value of offered act is 0 or 1
				   'notax':{'class':PWLDynamics,
						  'args':conditionalInc('self','negSatisfaction',0.3,[{'entity':'actor','feature':'offerednotax'},{'entity':'self','feature':'terminated'}],[])},

# cheeky place again
				   'tax':{'class':PWLDynamics,
					  'args':conditionalInc('self','negSatisfaction',-0.1,[{'entity':'actor','feature':'offerednotax'},{'entity':'self','feature':'terminated'}],[])},

				   'wait':{'class':PWLDynamics,
					   'args':conditionalInc('self','negSatisfaction',-0.1,[{'entity':'actor','feature':'offerednotax'},{'entity':'Market','feature':'tax'},{'entity':'self','feature':'terminated'}],[])}
				   }
		},

    'goals': USgoals,
    'models':{
##    'real':{'goals': USgoals,
##            'policy':[]
##            },

    # This model has the negotiator ask for no taxation, offers giving money, and follows through on that offer, if the taxation is removed
    'fair':
    {'goals':[{'entity':['Followers'],'direction':'max','type':'state','key':'money','weight':0.2},
              {'type':'actObject','key':'_message','entity':['self'],'direction':'max','weight':0.6},
              {'type':'actObject','key':'tax','entity':['ownMarket'],'direction':'min','weight':0.2},
              ],
     'policy':
     [
    # Request no taxation if not already offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'requestnotax','object':'Manager'}},
    # Offer money if requested
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','requestedgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','offeredgiveMoneyTo'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'offergiveMoneyTo','object':'Manager'}},
    # Accept if things match up
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','accepted'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'accept','object':'Manager'}},
    # Give money if offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','self','state','offeredgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','Market','state','tax'],
                 'range':Interval(-1.,0.001)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(0.5,1.)},
                ],
     'action':{'type':'giveMoneyTo','object':'Manager'}},
    # Otherwise, do nothing
    {'class':'default',
     'action':{'type':'doNothingTo','object':'Manager'},
     },
    ],
                         },
    # A model identical to 'fair', except that the agent will give the money regardless of whether the tax is lifted
    'weak':
    {'goals':[{'entity':['Followers'],'direction':'max','type':'state','key':'money','weight':0.2},
              {'type':'actObject','key':'_message','entity':['self'],'direction':'max','weight':0.6},
              {'type':'actObject','key':'tax','entity':['ownMarket'],'direction':'min','weight':0.2},
              ],
     'policy':
     [
    # Request no taxation if not already offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'requestnotax','object':'Manager'}},
    # Offer money if requested
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','requestedgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','offeredgiveMoneyTo'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'offergiveMoneyTo','object':'Manager'}},
    # Accept if things match up
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','accepted'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'accept','object':'Manager'}},
    # Give money if offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','self','state','offeredgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(0.5,1.)},
                ],
     'action':{'type':'giveMoneyTo','object':'Manager'}},
    # Otherwise, do nothing
    {'class':'default',
     'action':{'type':'doNothingTo','object':'Manager'},
     },
    ],
                         },
    'hardliner':
    {'goals':[{'entity':['Followers'],'direction':'max','type':'state','key':'money','weight':0.2},
              {'type':'actObject','key':'_message','entity':['self'],'direction':'max','weight':0.6},
              {'type':'actObject','key':'tax','entity':['ownMarket'],'direction':'min','weight':0.2},
              ],
     'policy':
     [
    # Request no taxation if not already offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'requestnotax','object':'Manager'}},
    # Accept if things match up
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','object','state','offerednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','requestednotax'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(-1.,0.5)},
                {'class':'belief',
                 'keys':['entities','self','state','accepted'],
                 'range':Interval(-1.,0.5)},
                ],
     'action':{'type':'accept','object':'Manager'}},
    # Give money if offered
    {'class':'conjunction',
     'clauses':[{'class':'belief',
                 'keys':['entities','self','state','offeredgiveMoneyTo'],
                 'range':Interval(0.5,1.)},
                {'class':'belief',
                 'keys':['entities','Market','state','tax'],
                 'range':Interval(-1.,0.001)},
                {'class':'belief',
                 'keys':['entities','self','state','terminated'],
                 'range':Interval(0.5,1.)},
                ],
     'action':{'type':'giveMoneyTo','object':'Manager'}},
    # Otherwise, do nothing
    {'class':'default',
     'action':{'type':'doNothingTo','object':'Manager'},
     },
    ],
     },
    },
##    'beliefs':{# 'Manager':{'model':'loyal'},
##               'Leader':{'_likeability':0.1,
##                         '_trustworthiness':0.1},
##               'Followers':{'_likeability':0.3,
##                            '_trustworthiness':0.3,
####                            'model':'loyal'
##                            },
##               },
    }    
