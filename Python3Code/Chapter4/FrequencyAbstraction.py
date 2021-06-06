##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import itertools

import numpy as np
import pandas as pd


# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:
    def __init__(self):
        self.temp_list = []
        self.sma_list = []
        self.freqs = None
        self.data_table = None
        self.periodic_predictor_cols = []
        self.window_size = 0

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses
    # the number of samples per second (i.e. Frequency is Hertz of the dataset).

    def compute_signal_magnitude_area(self, data, window_size, group):
        periodics = self.data_table.loc[data.index, group].copy()
        transformation = np.fft.rfft(periodics.T.to_numpy(), len(periodics))
        real_ampl = pd.DataFrame(transformation.real)
        # compute signal magnitude area
        sma = real_ampl.abs().sum().sum() / 3
        self.sma_list.append(sma)
        return 0

    def find_fft_transformation(self, data):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0 : len(real_ampl)])]
        # weigthed
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)
        # energy
        freq_energy = float(np.sum(real_ampl ** 2)) / len(data)

        # pse
        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        # Make sure there are no zeros.
        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        real_ampl = np.insert(real_ampl, 0, pse)
        real_ampl = np.insert(real_ampl, 0, freq_energy)

        self.temp_list.append(real_ampl)

        return 0

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, columns, window_size, sampling_rate):
        self.window_size = window_size
        self.data_table = data_table
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(self.window_size))).round(3)
        self.periodic_predictor_cols = itertools.chain(*columns.values())

        for group_name, group in columns.items():
            for col in group:
                collist = []
                # prepare column names
                collist.append(col + "_max_freq")
                collist.append(col + "_freq_weighted")
                collist.append(col + "_pse")
                collist.append(col + "_freq_energy")

                collist = collist + [
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(self.window_size)
                    for freq in self.freqs
                ]

                # rolling statistics to calculate frequencies, per window size.
                # Pandas Rolling method can only return one aggregation value.
                # Therefore values are not returned but stored in temp class variable 'temp_list'.

                # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
                self.data_table[col].rolling(self.window_size + 1).apply(
                    self.find_fft_transformation
                )

                # Pad the missing rows with nans
                frequencies = np.pad(
                    np.array(self.temp_list),
                    ((40, 0), (0, 0)),
                    "constant",
                    constant_values=np.nan,
                )
                # add new freq columns to frame

                self.data_table[collist] = pd.DataFrame(
                    frequencies, index=self.data_table.index
                )
                # reset temp-storage array
                del self.temp_list[:]

            # add the sma metric
            self.data_table[group[0]].rolling(self.window_size + 1).apply(
                self.compute_signal_magnitude_area,
                args=(
                    window_size,
                    group,
                ),
                raw=False,
            )

            sma_col_name = group_name + "_sma"
            self.data_table[sma_col_name] = np.nan
            self.data_table.iloc[
                40:, self.data_table.columns.get_loc(sma_col_name)
            ] = self.sma_list
            del self.sma_list[:]

        return self.data_table
