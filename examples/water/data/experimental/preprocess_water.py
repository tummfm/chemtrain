"""Helper file that creates interpolated datasets from json files of
WebplotDigitizer.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def load_fom_json(file, n_bincenters, curve_end, curve_start, init_offset,
                  delta_offset):
    with open(file, 'rt', encoding='utf-8') as f:
        json_data = json.load(f)

    gridpoints = np.linspace(0., curve_end, n_bincenters + 1)
    datasets = np.zeros((gridpoints.size, len(json_data['datasetColl'])))
    for i_dataset, dataset in enumerate(json_data['datasetColl']):
        offset = init_offset + i_dataset * delta_offset
        datapoints = dataset['data']
        data_array = np.zeros((len(dataset['data']) + 2, 2))

        for j, point in enumerate(datapoints):
            data_array[j + 1, :] = point['value']

        # set points beyond range to allow save interpolation
        data_array[0, :] = [-10., data_array[1, 1]]
        data_array[-1, :] = [curve_end + 10., data_array[-2, 1]]

        data_array[:, 1] -= offset  # drawing offset
        data_array[:, 1] = np.where(data_array[:, 0] >= curve_start,
                                    data_array[:, 1], 0.)
        interpolator = interpolate.interp1d(data_array[:, 0], data_array[:, 1],
                                            kind='cubic')
        interpolated = interpolator(gridpoints)

        datasets[:, i_dataset] = interpolated

    return np.hstack([np.expand_dims(gridpoints, -1), datasets])


temperatures = np.array([280., 288., 295., 313., 343., 365.])
rdf_data = load_fom_json('rdf_water_280K_365K.json', 300, 1., 0.22, 2., 1.)

adf_data = load_fom_json('adf_water_280K_365K.json', 150, 180., 40.,
                         0.003, 0.003)
# convert to rad and include sin dependence again
adf_data[:, 0] *= (np.pi / 180.)
sin_theta = np.expand_dims(np.sin(adf_data[:, 0]), -1)
adf_data[:, 1:] *= sin_theta

np.savetxt('temperatures.csv', temperatures)
np.savetxt('rdf_temperatures.csv', rdf_data)
np.savetxt('adf_temperatures.csv', adf_data)

plt.figure()
plt.ylabel('r in nm')
for i in range(1, rdf_data.shape[1]):
    plt.plot(rdf_data[:, 0], rdf_data[:, i], label=f'{temperatures[i - 1]} K')
plt.legend()
plt.savefig('rdf_over_temperatures.png')

plt.figure()
plt.ylabel('theta in deg')
for i in range(1, adf_data.shape[1]):
    plt.plot(adf_data[:, 0], adf_data[:, i], label=f'{temperatures[i - 1]} K')
plt.legend()
plt.savefig('adf_over_temperatures.png')






