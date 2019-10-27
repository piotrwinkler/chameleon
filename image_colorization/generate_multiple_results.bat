for %%x in ("V84", "V123", "V91", "V73", "V74", "V130") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V84", "V123", "V91", "V73", "V74", "V130") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
