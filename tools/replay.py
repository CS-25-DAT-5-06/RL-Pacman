import berkeley_pacman.pacman as pm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import berkeley_pacman.graphicsDisplay as graphicsDisplay

RECORDINGS_PATH = "data/recordings/"

def keyFunc(e):
    return e[0]


def replayGame(file):
    
    import pickle
    f = open(file, 'rb')
    try:
        fileDict = pickle.load(f)
    finally:
        f.close()
    fileDict['display'] = graphicsDisplay.PacmanGraphics(
    1.0, frameTime=0.1)

    import berkeley_pacman.pacmanAgents as pacmanAgents
    import berkeley_pacman.ghostAgents as ghostAgents
    rules = pm.ClassicGameRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i+1)
                                                for i in range(fileDict["layout"].getNumGhosts())]
    game = rules.newGame(fileDict["layout"],-1, agents[0], agents[1:], fileDict['display'])
    state = game.state
    fileDict['display'].initialize(state.data)

    for action in fileDict["actions"]:
            # Execute the action
        state = state.generateSuccessor(*action)
        # Change the display
        fileDict['display'].update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    fileDict['display'].finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog= "Replay Tool",
                description='Tool for replaying sessions or games of Pac-Man',
                )
    parser.add_argument("session", help="The name of the folder corresponding to the session that ")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-g","--game", help="Specific number corresponding to a game in the session")
    group.add_argument("-fl", "--firstandlast", action='store_true', help="Replays the first and last games")
    group.add_argument("-a","--all",action='store_true')
 
    args = parser.parse_args()
    
    
    
    sessionPath = RECORDINGS_PATH + args.session


    session = os.listdir(sessionPath)

    
    
    if(args.game != None):
        for recording in session:
            if(recording.split("-")[2] == args.game):
                print(f'Replaying {recording}.')
                replayGame(sessionPath + f"/{recording}")
                break
        print("Game does not exist")
    elif(args.all):
        for recording in session:
            print(f'Replaying {recording}.')
            replayGame(sessionPath + f"/{recording}")
    elif(args.firstandlast):
        list = []
        for recording in session:
            list.append((int(recording.split('-')[2]),recording))
            list.sort(key=keyFunc)
        print(list)
        print(f'Replaying {list[0][1]}.')
        replayGame(sessionPath +f"/{list[0][1]}")
        print(f'Replaying {list[len(list)-1][1]}.')
        replayGame(sessionPath + f"/{list[len(list)-1][1]}")

    
    # print('Replaying recorded game.')
    # import pickle
    # f = open(args["filename"], 'rb')
    # try:
    #     recorded = pickle.load(f)
    # finally:
    #     f.close()
    # recorded['display'] = graphicsDisplay.PacmanGraphics(
    # 1.0, frameTime=0.1)
    # replayGame(recorded)
    



