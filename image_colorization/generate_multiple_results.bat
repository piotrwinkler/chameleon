for %%x in ("V70", "V71", "V72", "V73", "V74") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V70", "V71", "V72", "V73", "V74") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
