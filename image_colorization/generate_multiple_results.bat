for %%x in ("V84", "V130", "V200") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V84", "V130", "V200") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
