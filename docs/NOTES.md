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

