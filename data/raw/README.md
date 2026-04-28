# Data Directory

Put each dataset's CSV files in its matching subdirectory:

- `data/raw/<dataset>/train.csv`
- `data/raw/<dataset>/test.csv`

Each CSV should include a `date` column. AlphaCast infers the target as the last
non-metadata data column.
