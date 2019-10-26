for %%x in ("V122") DO (python validate_chosen_trainings.py -v %%x)

for %%x in ("V122") DO (ECHO Choosing version %%x & python deploy_single_training.py -v %%x)
