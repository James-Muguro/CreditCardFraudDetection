import pandera as pa
from pandera import Column, Check, DataFrameSchema

# The dataset has columns: Time, V1..V28 (PCA components), Amount, Class
pca_columns = {
    f"V{i}": Column(float, nullable=False) for i in range(1, 29)
}

credit_card_schema = DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), nullable=False),
        **pca_columns,
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,   # fail if unexpected extra columns appear
    coerce=True,   # attempt safe dtype coercion before validating
)