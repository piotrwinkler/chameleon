for %%x in ("V131", "V132", "V133", "V134") DO (python validate_chosen_trainings.py -v %%x)

for %%x in ("V131", "V132", "V133", "V134") DO (ECHO Choosing version %%x & python deploy_single_training.py -v %%x)
