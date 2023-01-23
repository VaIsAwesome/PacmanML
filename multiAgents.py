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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        
        #my variables
        currentfoodList = currentGameState.getFood().asList()
        nextfoodList = newFood.asList()
        ghostLocations = []

        # get ghost positions
        for ghost in newGhostStates:
            ghostLocations.append((ghost.getPosition()[0], ghost.getPosition()[1]))


        #Worst Case: Ghost is in the successive state -->  Return worst possible val
        if newPos in ghostLocations and newScaredTimes[0] <= 0:
            return float("-inf")


        #Best Case: There is food in the successive state --> Return best possible val
        elif newPos in currentfoodList:
            return float("inf")

        #Second Best: Find successor that goes farther from ghosts and closer to foods.
        #How?: Find the closest food and the closest ghost. We want closest food and not closest ghost.
        else:
            distanceSort = lambda x : util.manhattanDistance(x, newPos)
            closestFood = sorted(nextfoodList, key = distanceSort)[0]
            closestGhost = sorted(ghostLocations, key = distanceSort)[0]
    
            return  1/distanceSort(closestFood) - 1/distanceSort(closestGhost)

        
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        #Important Variables
        GhostIndex = range(1,gameState.getNumAgents())
        depth = self.depth
        PacManMoves = gameState.getLegalActions(0)

        #finding max values
        def get_max(state, curr_depth):

            #check if deepest gamestate explored
            if ((state.isLose() or state.isWin()) or curr_depth == depth):
                return self.evaluationFunction(state)

            #determine max value from all of PacMan's legal actions
            else:
                tmp_max = float("-inf")
                PacMan_moves = state.getLegalActions(0)
                for action in PacMan_moves:
                    tmp_max = max(get_min(state.generateSuccessor(0, action), curr_depth, 1), tmp_max)
                return tmp_max


        #finding min values
        def get_min(state, curr_depth, ghost):

            #check if deepest gamestate explored
            if (state.isLose() or state.isWin() or curr_depth == depth):
                return self.evaluationFunction(state)
            
            #determine min value from all of ghost's moves
            tmp_min = float("inf")
            ghost_moves = state.getLegalActions(ghost)

            #not the last ghost, get min-min
            if ghost != GhostIndex[-1]:
                for action in ghost_moves:
                    successorMin = get_min(state.generateSuccessor(ghost, action), curr_depth, ghost + 1)
                    tmp_min = min(successorMin, tmp_min)
            
            #is the last ghost, get min-max
            else:
                for action in ghost_moves:
                    successorMax = get_max(state.generateSuccessor(ghost, action), curr_depth + 1)
                    tmp_min = min(successorMax, tmp_min)

            return tmp_min

        
        PacManMoves.sort(key = lambda x : get_min(gameState.generateSuccessor(0,x),0,1), reverse=True)

        return PacManMoves[0] #returns min

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #Important variables
        PacManMoves = gameState.getLegalActions(0)
        alpha, beta = float("-inf"), float("inf")
        tmpMove = None

        def getVal(state, alpha, beta, agentId, depth=0):
            agentId = agentId % state.getNumAgents()

            
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            #alpha beta pruning
            #if PacMan
            elif agentId == 0:
                agentMoves = state.getLegalActions(agentId)
                score = float("-inf")
                for move in agentMoves:
                    next_state = state.generateSuccessor(agentId, move)
                    score = max(score, getVal(next_state, alpha, beta, agentId + 1, depth - 1)) 
                
                    if score > beta:
                        return score
                    alpha = max(alpha, score)
                return score
            
            #if Ghost
            else:
                #print('pruning for ghost')
                agentMoves = state.getLegalActions(agentId)
                score = float("inf")
                for move in agentMoves:
                    next_state = state.generateSuccessor(agentId, move)
                    score = min(score, getVal(next_state, alpha, beta, agentId + 1, depth - 1)) 
                    
                    if score < alpha:
                        return score
                    beta = min(beta, score)
                return score

        # actual loop
        for move in PacManMoves:
            state = gameState.generateSuccessor(0, move)
            depth = self.depth * gameState.getNumAgents() - 1
            score = getVal(state, alpha, beta, 1, depth)
            if score > alpha:
                alpha, tmpMove = score, move
        return tmpMove


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        #Important Variables
        PacManMoves = gameState.getLegalActions(0)
        allScores = []
        depth = self.depth * gameState.getNumAgents() - 1

        def getValue(state, agentId, depth):
            agentId = agentId % state.getNumAgents()
            
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)
            
            if agentId == 0:
                moves = state.getLegalActions(agentId)
                prob = 1 / len(moves)
                score = float("-inf")

                for action in moves:
                    next = state.generateSuccessor(agentId, action)    
                    score = max(getValue(next, agentId + 1, depth - 1), score)  
                return score
            else:
                moves = state.getLegalActions(agentId)
                prob = 1 / len(moves)
                score = 0

                for action in moves:
                    next = state.generateSuccessor(agentId, action)     
                    score += prob * getValue(next, agentId + 1, depth - 1)
                    
                return score        
        
        for i, action in enumerate(PacManMoves):
            allScores.append(( i, getValue(gameState.generateSuccessor(0, action), 1, depth) ))

        topScore = sorted(allScores, key = lambda x: x[1], reverse=True)[0]        
        return PacManMoves[topScore[0]]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Creates a score which is higher for squares closer to scared
    ghosts and lower for normal ghosts, also is higher for food squares. Scores are scaled by distances of these factors. 
    """
    "*** YOUR CODE HERE ***"

    #important variables
    pacPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # Find closest food distance
    distToFoodList = []
    for foodPos in currentGameState.getFood().asList():
        distToFoodList.append(util.manhattanDistance(pacPos, foodPos))
    if len(distToFoodList) != 0:
        score += 1/min(distToFoodList)
    else:
        score += 1

    # Find closest ghosts
    for ghost in currentGameState.getGhostStates():
        dist = manhattanDistance(pacPos, ghost.getPosition())
        if dist == 0:
            return float('-inf')

        if ghost.scaredTimer < 1:
            score += -1/dist
        else:
            score += 2/dist

    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
