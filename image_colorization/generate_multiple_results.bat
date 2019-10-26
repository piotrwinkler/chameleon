for %%x in ("V74", "V110", "V111", "V112", "V113") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V74", "V110", "V111", "V112", "V113") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
