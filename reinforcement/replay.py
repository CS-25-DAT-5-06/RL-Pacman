import pacman as pm
import sys

if __name__ == "__main__":
    import graphicsDisplay
    
    filename = sys.argv[1]

    print('Replaying recorded game.')
    import pickle
    f = open(filename, 'rb')
    try:
        recorded = pickle.load(f)
    finally:
        f.close()
    recorded['display'] = graphicsDisplay.PacmanGraphics(
    1.0, frameTime=0.1)

    import pacmanAgents
    import ghostAgents
    rules = pm.ClassicGameRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i+1)
                                             for i in range(recorded["layout"].getNumGhosts())]
    game = rules.newGame(recorded["layout"],-1, agents[0], agents[1:], recorded['display'])
    state = game.state
    recorded['display'].initialize(state.data)

    for action in recorded["actions"]:
            # Execute the action
        state = state.generateSuccessor(*action)
        # Change the display
        recorded['display'].update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    recorded['display'].finish()
