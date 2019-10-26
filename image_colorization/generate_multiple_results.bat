for %%x in ("V73", "V100", "V101", "V102", "V103") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V73", "V100", "V101", "V102", "V103") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
