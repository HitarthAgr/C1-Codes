import requests
import pandas as pd
import matplotlib.pyplot as plt

# API key and location coordinates
api_key = "1f87700e618acc5cf4f567a0f071b329" # USe your own api key
lat = 18.184135
lon = 74.610764

# Construct API URL and send request
# Use pure numbers without 'N' or 'E'
lat = 26.850000
lon = 80.950000
api_key = "1f87700e618acc5cf4f567a0f071b329"

# Construct API URL correctly
url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
data = response.json()

# Check if the API request was successful (Status code 200)
if response.status_code == 200:
    # Extract weather information
    temperatures = [x['main']['temp'] for x in data['list']]
    humidity = [x['main']['humidity'] for x in data['list']]
    wind_speed = [x['wind']['speed'] for x in data['list']]
    timestamps = [pd.to_datetime(x['dt'], unit='s') for x in data['list']]

    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature': temperatures,
        'Humidity': humidity,
        'Wind Speed': wind_speed
    })

    # Set timestamp as index
    df.set_index('Timestamp', inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Convert Kelvin into Celsius
    df['Temperature'] = df['Temperature'] - 273.15

    # Display maximum and minimum temperature
    print("Maximum Temperature:", df['Temperature'].max())
    print("Minimum Temperature:", df['Temperature'].min())

    print(df)

    # Plot temperature
    df['Temperature'].plot()
    plt.title("Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (Celsius)")
    plt.show()
else:
    # Print the error message returned by OpenWeatherMap
    print(f"Failed to fetch data. Status Code: {response.status_code}")
    print("Error Message:", data.get("message", "Unknown error"))