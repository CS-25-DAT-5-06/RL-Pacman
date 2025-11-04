from gymenv import PacManEnv
import gymnasium as gym
import pacman as pm
import sys

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    args = pm.readCommand(sys.argv[1:])  # Get game components based on input

    gym.register(id="berkley-pacman",entry_point=PacManEnv,max_episode_steps=300,kwargs = {"args": args})
    env = gym.make("berkley-pacman")
  
    

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
