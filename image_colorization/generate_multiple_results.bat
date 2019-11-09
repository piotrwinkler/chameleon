for %%x in ("V84", "V130", "V140", "V150") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V84", "V130", "V140", "V150") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
