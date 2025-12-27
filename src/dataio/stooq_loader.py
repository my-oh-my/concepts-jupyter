import io

import pandas as pd
import requests


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Define the URL for the HTTP GET request
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"

    # Send the GET request to the URL
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return pd.DataFrame()

    # Read the CSV content from the response
    csv_data = response.content.decode("utf-8")

    # Use io.StringIO to convert the string data into a file-like object
    csv_file_like = io.StringIO(csv_data)

    # Load the CSV data into a Pandas DataFrame with 'Date' as the index
    data = pd.read_csv(csv_file_like, index_col="Date", parse_dates=True)

    # Display the first few rows of the DataFrame
    return data.loc[start_date:end_date]  # type: ignore[misc]
