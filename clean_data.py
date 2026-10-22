import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Combine relevant text fields into one column for the model to learn from
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
for col in text_columns:
    df[col] = df[col].fillna("")

df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Keep only what we need: the combined text and the label
cleaned_df = df[["text", "fraudulent"]]

# Drop any rows where text ended up empty
cleaned_df = cleaned_df[cleaned_df["text"].str.strip() != ""]

# Save cleaned data
cleaned_df.to_csv("data/cleaned_data.csv", index=False)

print("Cleaning complete.")
print("Shape of cleaned data:", cleaned_df.shape)
print(cleaned_df["fraudulent"].value_counts())