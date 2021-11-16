# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            tmpvalues = util.Counter()
            for j in self.mdp.getStates():
                tmpmax = []
                if self.mdp.isTerminal(j):
                    tmpvalues[j] = self.mdp.getReward(j,'exit','')
                else:
                    for k in self.mdp.getPossibleActions(j):
                        tmpmax.append(self.getQValue(j,k))
                    tmpvalues[j] = max(tmpmax)
            self.values = tmpvalues





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextpossible = self.mdp.getTransitionStatesAndProbs(state,action)
        Qvalue = 0
        
        for s, p in nextpossible:
            Qvalue = Qvalue + p*(self.discount*self.values[s]+self.mdp.getReward(state,action,s))
        
        return Qvalue


        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        nextaction = self.mdp.getPossibleActions(state)

        if not len(nextaction):
            return None

        tmp = []
        for a in nextaction:
            tmp.append((a,self.getQValue(state,a)))
        
        return max(tmp,key = lambda x:x[1])[0]


        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.values = util.Counter()
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i%len(states)]
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getQValue(state,self.getAction(state))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.values = util.Counter()
        states = self.mdp.getStates()
        Prique = util.PriorityQueue()
        diff = 0

        for s in states:
            if not self.mdp.isTerminal(s):
                diff = abs(self.values[s]-self.highest_Q(s))
                Prique.push(s,-diff)
        
        for iter in range(self.iterations):

            if Prique.isEmpty():
                return
            
            s = Prique.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = self.highest_Q(s)
            
            for p in self.predecessor(s):
                diff = abs(self.values[p]-self.highest_Q(p))
                if diff > self.theta:
                    Prique.update(p,-diff)

    def highest_Q(self,state):
        high = -10**10
        for i in self.mdp.getPossibleActions(state):
            if high < self.getQValue(state,i):
                high = self.getQValue(state,i)
            else:
                high = high
        return high

    def predecessor(self,state):
        pred = set()
        states =  self.mdp.getStates()
        
        if not self.mdp.isTerminal(state):
            for s in states:
                if not self.mdp.isTerminal(s):
                    for a in self.mdp.getPossibleActions(s):
                        for i,j in self.mdp.getTransitionStatesAndProbs(s,a):
                            if (i == state) and (j>0):
                                pred.add(s)
        return pred