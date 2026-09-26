import pandas as pd

print("Loading original test IDs...")
# Read just the first column (the IDs) from the original test set
test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', usecols=[0])
id_col = test_s1.columns[0]

print("Loading model predictions...")
preds = pd.read_csv('output/matching_results.tsv', sep='\t', dtype=str)

print("Filling in the blanks for missing IDs...")
# Merge them so every single original ID is present, filling missing ones with blanks
final_submission = test_s1.merge(preds, on=id_col, how='left').fillna("")

final_submission.to_csv('output/final_submission.tsv', sep='\t', index=False)
print("✅ Fixed! Your perfect submission is ready at output/final_submission.tsv")
