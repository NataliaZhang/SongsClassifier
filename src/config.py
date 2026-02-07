from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    sample_sub_csv: str = "sample_submission.csv"
    output_dir: str = "outputs"
    submission_csv: str = "submission.csv"

TARGET_COL = "Popularity_Type"