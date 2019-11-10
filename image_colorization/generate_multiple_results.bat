for %%x in ("V70", "V84", "V86", "V87") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V70", "V84", "V86", "V87") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
