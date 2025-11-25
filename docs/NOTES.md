24/11/PETER:

Difference between feature engineering and heuristics:
State abstraction is feature engineering, not heuristics.
Vi tager observation space, og fortolker dette, og giver dette "optimerede" state space til agenten.
Ved heuristics, siger vi direkte, hvad der er godt og dårligt

Mangler: 
- Fjerne STOP fra gymenv_v2.py impl. 
- Mangler helper functions i state abstraction + feature opsætning (50% færdig)
- Inkluder qtable størrelse i træningsdata/output
- Træning med state abstraction, kan træne på Classic layout?
- Fjerne vores egne filer fra reinforcement folder
- Omdøb reinforcement folder til berkeley pacman, se readme.md

25/11 - Nikolaj
For at køre filer skal man nu skrive "python -m <package>.<module>" et eksempel er  "python -m environement.gymenv_v2". Derudover skal alle import statements også indholde en reference til den pakke filen er i inklusiv filer i den folder den tilhører(er ikke helt sikker på at dette gælder men det fortæller min erfaring indtil videre i hvertfald)