for %%x in ("V112", "V113") DO (python validate_chosen_trainings.py -v %%x)

for %%x in ("V112", "V113") DO (ECHO Choosing version %%x & python deploy_single_training.py -v %%x)
