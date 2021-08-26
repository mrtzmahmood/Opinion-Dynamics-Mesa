import numpy as np
import pandas as pd
import colorsys
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from .agent import odAgent
import statistics

class odModel(Model):
    def __init__(self, N, min_eps, max_eps, alpha, beta, cr, agg, ee_rate, ext_range, ext_type, org, width, height, max_iters):       
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.alpha = alpha
        self.beta = beta
        self.num_agents = N
        self.communication_regime = cr
        self.aggregation_in_HK = agg
        self.entry_exit_rate = ee_rate
        self.exteremisim_range = ext_range
        self.exteremisim_type = ext_type
        self.original = org
        self.space = ContinuousSpace(width, height, True, 0, 0)
        self.schedule = RandomActivation(self)
        self.max_iters = max_iters
        self.iteration = 0  
        self.running = True
        model_reporters = {
            'Opinion mean': lambda m: self.rpt_opinion_mean (m),
            'Opinion median': lambda m: self.rpt_opinion_median (m)  
        }        
        agent_reporters = {
            "x": lambda a: a.pos[0],
            "y": lambda a: a.pos[1],
            "opinion": lambda a: a.opinion,
            'eps': lambda a: a.eps
        }
        self.dc = DataCollector(model_reporters=model_reporters,
                                agent_reporters=agent_reporters)  
        # Create agents
        for i in range(self.num_agents):
            x = 0
            y = 0
            pos = np.array((x, y))
            oda = odAgent(i, self)
            self.space.place_agent(oda, oda.pos)
            self.schedule.add(oda)

    def step(self):
        self.dc.collect(self) 
        self.schedule.step()
        if self.communication_regime == "DW" and self.original:
            for agent in range (len(self.schedule.agents)):
                other = agent.random.choice(self.schedule.agents)
                other.update_opinion()
        else:
            for agent in self.schedule.agents:
                agent.update_opinion()
        for agent in self.schedule.agents:             
            loc = len(agent.opinion_list) - 1
            agent.opinion_list[loc] = agent.opinion
        for agent in self.schedule.agents: 
            if len (agent.opinion_list) == self.space.x_max + 1:
                agent.opinion_list.pop(0)
        for agent in self.schedule.agents:                     
            agent.entry_exit()  
        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False    
    
    @staticmethod        
    def rpt_opinion_mean (model):
        opinions = []
        for agent in model.schedule.agents:
            opinions.append(agent.opinion)
        opinion_mean = statistics.mean(opinions)  
        return opinion_mean

    @staticmethod        
    def rpt_opinion_median (model):
        opinions = []
        for agent in model.schedule.agents:
            opinions.append(agent.opinion)
        opinion_median = statistics.median(opinions)  
        return opinion_median
