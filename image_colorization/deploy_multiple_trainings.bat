for %%x in ("V74", "V90", "V91", "V92", "V93", "V100", "V101", "V102", "V103") DO (python validate_chosen_trainings.py -v %%x)

for %%x in ("V74", "V90", "V91", "V92", "V93", "V100", "V101", "V102", "V103") DO (ECHO Choosing version %%x & python deploy_single_training.py -v %%x)
