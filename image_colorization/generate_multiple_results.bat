for %%x in ("V123", "V131", "V91", "V132", "V101", "V133", "V74", "V134") DO (python validate_chosen_testers.py -v %%x)

for %%x in ("V123", "V131", "V91", "V132", "V101", "V133", "V74", "V134") DO (ECHO Choosing version %%x & python generate_single_results.py -v %%x)
