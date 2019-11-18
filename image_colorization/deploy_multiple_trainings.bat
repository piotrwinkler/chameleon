for %%x in ("V170", "V171", "V172", "V173") DO (python validate_chosen_trainings.py -v %%x)

for %%x in ("V170", "V171", "V172", "V173") DO (ECHO Choosing version %%x & python deploy_single_training.py -v %%x)
