import pandas as pd

# Load the CSV
df = pd.read_csv("NTPC.csv")

# Define a function to convert dates correctly
def convert_date(date_str):
    possible_formats = ["%d-%m-%Y", "%d/%m/%Y", "%-d/%-m/%Y", "%Y-%m-%d"]  # Add more if needed

    for fmt in possible_formats:
        try:
            return pd.to_datetime(date_str, format=fmt).strftime("%-d/%-m/%Y")
        except ValueError:
            continue  # Try the next format

    return None  # Return None if all formats fail (keeps track of errors)

# Apply conversion to the date column
df["date"] = df["date"].astype(str).apply(convert_date)

# Drop rows where date conversion failed (optional)
df = df.dropna(subset=["date"])

# Save the cleaned CSV
df.to_csv("your_file_cleaned.csv", index=False)
