# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        #food distance
        foodlist = newFood.asList()
        foodis = [10**10]
        for f in foodlist:
            foodis.append(manhattanDistance(newPos, f))
    
        minfood = min(foodis)
        # ghost distance
        ghostdis = [10**10]
        for g in successorGameState.getGhostPositions():
            ghostdis.append(manhattanDistance(newPos,g))
        
        mingh = min(ghostdis)
        #function in terms of ghostdis and fooddis
        epsilon = 10**-10
        index = 1-1/(1000*mingh+epsilon)
        score = score + mingh*index/minfood
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.max_val(gameState, 0, 0)[0]

    

    def max_val(self, gameState, agent, depth):
        action = [("action",-10**10)]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                action.append((j,score))
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth)[1]
                action.append((j,score))
                continue
            if next_agent:
                score = self.min_val(new_state,next_agent,depth)[1]
                action.append((j,score))
                continue

        action = max(action, key=lambda x:x[1])
        return action
    def min_val(self, gameState, agent, depth):
        action = [("action",10**10)]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                action.append((j,score))
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth)[1]
                action.append((j,score))
                continue
            if next_agent:
                score = self.min_val(new_state,next_agent,depth)[1]
                action.append((j,score))
                continue

        action = min(action, key=lambda x:x[1])
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_val(gameState, 0, 0, -10**10, 10**10)[0]

    def max_val(self, gameState, agent, depth,alpha,beta):
        action = [("action",-10**10)]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                action.append((j,score))
                if max(action, key = lambda x:x[1])[1]>beta:
                    return max(action, key = lambda x:x[1])
                alpha = max(alpha,max(action, key = lambda x:x[1])[1])
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth,alpha,beta)[1]
                action.append((j,score))
                if max(action, key = lambda x:x[1])[1]>beta:
                    return max(action, key = lambda x:x[1])
                alpha = max(alpha,max(action, key = lambda x:x[1])[1])
                continue
            if next_agent:
                score = self.min_val(new_state,next_agent,depth,alpha,beta)[1]
                action.append((j,score))
                if max(action, key = lambda x:x[1])[1]>beta:
                    return max(action, key = lambda x:x[1])
                alpha = max(alpha,max(action, key = lambda x:x[1])[1])
                continue

        action = max(action, key=lambda x:x[1])
        return action
    
        
    def min_val(self, gameState, agent, depth,alpha,beta):
        action = [("action",10**10)]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                action.append((j,score))
                if min(action, key = lambda x:x[1])[1]<alpha:
                    return min(action, key = lambda x:x[1])
                beta = min(beta,min(action, key = lambda x:x[1])[1])
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth,alpha,beta)[1]
                action.append((j,score))
                if min(action, key = lambda x:x[1])[1]<alpha:
                    return min(action, key = lambda x:x[1])
                beta = min(beta,min(action, key = lambda x:x[1])[1])
                continue
            if next_agent:
                score = self.min_val(new_state,next_agent,depth,alpha,beta)[1]
                action.append((j,score))
                if min(action, key = lambda x:x[1])[1]<alpha:
                    return min(action, key = lambda x:x[1])
                beta = min(beta,min(action, key = lambda x:x[1])[1])
                continue

        action = min(action, key=lambda x:x[1])
        
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.

          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # calling expectimax with the depth we are going to investigate
        return self.max_val(gameState, 0, 0)[0]

    

    def max_val(self, gameState, agent, depth):
        action = [("action",-10**10)]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                action.append((j,score))
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth)[1]
                action.append((j,score))
                continue
            if next_agent:
                score = self.mean_val(new_state,next_agent,depth)
                action.append((j,score))
                continue

        action = max(action, key=lambda x:x[1])
        return action
    def mean_val(self, gameState, agent, depth):
        value = [0]
        score = 0
        depth = depth + 1

        for j in gameState.getLegalActions(agent):
            new_state = gameState.generateSuccessor(agent,j)
            next_agent = depth%new_state.getNumAgents()

            if depth is self.depth * new_state.getNumAgents() or new_state.isLose() or new_state.isWin():
                score = self.evaluationFunction(new_state)
                value.append(score)
                continue
            if next_agent == 0:
                score = self.max_val(new_state,next_agent,depth)[1]
                value.append(score)
                continue
            if next_agent:
                score = self.mean_val(new_state,next_agent,depth)
                value.append(score) 
                continue

        value = sum(value)/(len(value)-1)
        return value

import math

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsule = currentGameState.getCapsules()
    score = currentGameState.getScore()

    #food score
    foodlist = newFood.asList()
    foodis = [10**10]
    for f in foodlist:
        foodis.append(manhattanDistance(newPos, f))
    
    minfood = min(foodis)
    score += 10.0/minfood

    #capsule score
  
    capsuledis = [10**10]
    for g in newCapsule:
        capsuledis.append(manhattanDistance(newPos,g))

    mincapsule = min(capsuledis)
    score +=100.0/mincapsule

    #ghost score
    ghostdis = [("ghost",10**10)]
    for k in newGhostStates:
        tmp = manhattanDistance(newPos,k.getPosition())
        ghostdis.append((k,tmp))
    min_ghost = min(ghostdis,key = lambda x:x[1])
    if min_ghost[0].scaredTimer>0:
        score += 5000.0/(min_ghost[1]+10**-10)
    else:
        score -=1.0/(min_ghost[1]+10**-10)


    return score


# Abbreviation
better = betterEvaluationFunction

