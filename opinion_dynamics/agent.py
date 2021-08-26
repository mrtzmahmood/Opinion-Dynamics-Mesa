import numpy as np
import pandas as pd
import colorsys
from mesa import Agent

class odAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = self.new_opinion()
        self.opinion_list = [] #list (self.opinion)
        self.exteremist = False
        x = 0
        y = self.opinion * self.model.space.y_max
        pos = np.array((x, y))
        self.pos = pos
#        self.eps , self.color = self.new_confidence_bounds()
        self.eps  = self.new_confidence_bounds()
        

    def new_confidence_bounds(self):
        x = np.random.gamma(self.model.alpha, 1, 1)
        eps = x[0] / (x[0] + np.random.gamma(self.model.beta, 1, 1))
        eps = self.model.min_eps + (eps[0]*(self.model.max_eps - self.model.min_eps))
        #color = self.colorcode(eps, 0.5)
        return eps #, color 
    
    def colorcode (self, x, max_x):
        return colorsys.rgb_to_hsv ((190 - 190*(x/max_x)), 255, 255) 
    
    def new_opinion(self):
        return np.random.random()
    
    def entry_exit(self):
        if np.random.random() < self.model.entry_exit_rate :
            self.opinion = self.new_opinion()
            
    def aggregate (self, opinions):
        if self.model.aggregation_in_HK == "mean":
            mean= np.mean(opinions)
            return mean
        else: # median
            median = np.median(opinions)          
            return median
        
    def update_opinion(self):
        if self.model.exteremisim_type == "two side":
            if (0.5 - abs(self.opinion - 0.5)) < self.model.exteremisim_range :
                self.exteremist = True
        else: # exteremist type = one side
            if self.opinion  < self.model.exteremisim_range :
                self.exteremist = True
        if self.model.communication_regime == "DW":
            partner = self.random.choice(self.model.schedule.agents)
            if (abs(self.opinion - partner.opinion) < self.eps) and not self.exteremist:
                self.opinion = (self.opinion + partner.opinion) /2
                if self.model.original:
                    if self.model.exteremisim_type == "two side":
                        if (0.5 - abs(partner.opinion - 0.5)) < self.model.exteremisim_range :
                            partner.exteremist = True
                    else:
                        if partner.opinion  < self.model.exteremisim_range :
                            partner.exteremist = True
                    if (abs(partner.opinion - self.opinion) < partner.eps) and not partner.exteremist:
                        partner.opinion = (partner.opinion + self.opinion) /2
                        
        else: #communication type = HK
            if not self.exteremist:
                opt=[]
                if self.model.original:
                    for a in self.model.schedule.agents:
                        opt.append (a.opinion_list[len(a.opinion_list)-1])
                    fopt = list(filter(lambda o: abs(o - self.opinion_list[len(self.opinion_list)-1]) < self.eps, opt))
                else:
                    for a in self.model.schedule.agents:
                        opt.append (a.opinion)                    
                    fopt = list(filter(lambda o: abs(o - self.opinion) < self.eps, opt))
                self.opinion = self.aggregate(fopt)
                    
                        
    def step(self):
        self.opinion_list.append(self.opinion)

