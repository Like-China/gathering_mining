"""
Change the Porto timestamp to a random value
"""
import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd
import settings

raw_f = h5py.File("E:/data/porto.h5", 'r')
new_f = h5py.File("E:/data/porto1.h5", 'a')
num = raw_f.attrs['num']

count = 0
for ii in tqdm(range(num)):
    # Restricted latitude and longitude
    trip = np.array(raw_f.get('trips/' + str(ii+1)))
    trip = pd.DataFrame(trip, columns=['lon', 'lat'])
    trip = trip[trip.lon >= settings.lons_range_pt[0]]
    trip = trip[trip.lon <= settings.lons_range_pt[1]]
    trip = trip[trip.lat >= settings.lats_range_pt[0]]
    trip = trip[trip.lat <= settings.lats_range_pt[1]]
    trip = np.array(trip)
    trips = []
    while len(trip) >= settings.max_len:
        # Generate a trajectory of random length according to the length of min-max trajectory
        rand_len = np.random.randint(settings.min_len, settings.max_len)
        trips.append(trip[0:rand_len])
        trip = trip[rand_len:]
    if len(trip) >= settings.min_len:
        trips.append(trip)

    for trip in trips:
        length = len(trip)
        # Create a random timestamp
        random_timestamp = np.random.randint(0, 60000)
        timestamps = [random_timestamp + 5*i for i in range(length)]
        new_f["trips/" + str(count)] = trip
        new_f["timestamps/" + str(count)] = timestamps
        count += 1
new_f.attrs['num'] = count + 1
print("Write a valid trajectory: ", count + 1)
raw_f.close()
new_f.close()

# Test the timestamp generated by the simulation
new_f = h5py.File("E:/data/porto1.h5", 'r')
print(new_f.attrs['num'])
ts = np.array(new_f.get('timestamps/' + str(2)))
print(ts)