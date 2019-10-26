for %%x in ("V71", "V120", "V121", "V122", "V123") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V71", "V120", "V121", "V122", "V123") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
