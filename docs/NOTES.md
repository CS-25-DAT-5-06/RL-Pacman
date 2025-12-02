2/12/PETER:
NOTE: Which speed is pacman training at, when not rendering? Any optimizations?
Hook up to tensorboard is easy
Add recording flags in the experiment_runner and configuration files.


Reg. experiments:

mediumClassic:

No abstraction:             1000 episodes:  55000 states
                            2000 episodes: 106000 states

Simple abstraction:         2000 episodes:  35000 states
                            3000 episodes:  46000 states

Medium abstraction:         2000 episodes:  35000 states
                            3000 episodes:  48000 states

Relative distance:          2000 episodes:   3577 states
+ 1 ghost pos               3000 episodes:   3900 states
                            5000 episodes:   4118 states

Relative distance           2000 episodes:   3700 states
+ ghost relative pos:       3000 episodes:   4600 states   
                            5000 episodes:   5900 states

Relative distance
+ ghost grid relative:      2000 episodes:   2800 states
                            3000 episdes:    3300 states

Crisis mode:                2000 episodes:   201 states
                            3000 episodes:   204 states


Crisis +  BFS:              2000 episodes:   609 states
                            3000 episodes:   618 states #0.99 winrate!



25/11/PETER:
Note: The argmax function is ´not random, will always pick the first value, if q_values are equal. Exploration bias. Could be made better in choosing best q-value: qlearning_agent.py

24/11/PETER:
Difference between feature engineering and heuristics:
State abstraction is feature engineering, not heuristics.
Vi tager observation space, og fortolker dette, og giver dette "optimerede" state/observation space til agenten.
Ved heuristics, siger vi direkte, hvad der er godt og dårligt
Manhattan distance measures the sum of the absolute differences between the coordinates of the points. Ikke korteste afstand, men tager heller ikke højde for vægge etc.


Mangler: 
- Fjerne STOP fra gymenv_v2.py impl. 
- Mangler helper functions i state abstraction + feature opsætning (50% færdig)
- Inkluder qtable størrelse i træningsdata/output
- Træning med state abstraction, kan træne på Classic layout?
- Fjerne vores egne filer fra reinforcement folder
- Omdøb reinforcement folder til berkeley pacman, se readme.md

