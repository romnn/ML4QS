##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import CategoricalAbstraction, NumericalAbstraction
from Chapter4.TextAbstraction import TextAbstraction
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path("./intermediate_datafiles/")
DATASET_FNAME = "chapter3_result_final.csv"
RESULT_FNAME = "chapter4_result.csv"


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + " : " + str(value))


def main():
    print_flags()

    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, nrows=None, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print("File not found, try to run previous crowdsignals scripts first!")
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (
        dataset.index[1] - dataset.index[0]
    ).microseconds / 1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == "aggregation":
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [
            int(float(5000) / milliseconds_per_instance),
            int(float(0.5 * 60000) / milliseconds_per_instance),
            int(float(5 * 60000) / milliseconds_per_instance),
        ]

        # please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.

        for ws in window_sizes:

            dataset = NumAbs.abstract_numerical(dataset, ["acc_phone_x"], ws, "mean")
            dataset = NumAbs.abstract_numerical(dataset, ["acc_phone_x"], ws, "std")

        DataViz.plot_dataset(
            dataset,
            ["acc_phone_x", "acc_phone_x_temp_mean", "acc_phone_x_temp_std", "label"],
            ["exact", "like", "like", "like"],
            ["line", "line", "line", "points"],
        )
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == "frequency":
        # Now we move to the frequency domain, with the same window size.

        fs = float(1000) / milliseconds_per_instance
        ws = int(float(10000) / milliseconds_per_instance)
        dataset = FreqAbs.abstract_frequency(dataset, ["acc_phone_x"], ws, fs)

        # Spectral analysis.
        DataViz.plot_dataset(
            dataset,
            [
                "acc_phone_x_max_freq",
                "acc_phone_x_freq_weighted",
                "acc_phone_x_pse",
                "label",
            ],
            ["like", "like", "like", "like"],
            ["line", "line", "line", "points"],
        )
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == "final":

        ws = int(float(0.5 * 60000) / milliseconds_per_instance)
        fs = float(1000) / milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not "label" in c]

        numerical_aggregations = [
            (selected_predictor_cols, ws, "mean"),
            (selected_predictor_cols, ws, "std"),
            # TODO: Add your own aggregation methods here
            (selected_predictor_cols, ws, "mad"),
            (selected_predictor_cols, ws, "sum"),
        ]

        for _cols, _ws, _agg in numerical_aggregations:
            print(
                "computing %s for %d columns (window size %d) ..."
                % (_agg, len(_cols), _ws)
            )
            dataset = NumAbs.abstract_numerical(dataset, _cols, _ws, _agg)

        DataViz.plot_dataset(
            dataset,
            [
                "acc_phone_x",
                "gyr_phone_x",
                "hr_watch_rate",
                "light_phone_lux",
                "mag_phone_x",
                "press_phone_",
                "pca_1",
                "label",
            ],
            ["like", "like", "like", "like", "like", "like", "like", "like"],
            ["line", "line", "line", "line", "line", "line", "line", "points"],
        )

        CatAbs = CategoricalAbstraction()

        print("computing categorical abstractions ...")
        dataset = CatAbs.abstract_categorical(
            dataset,
            ["label"],
            ["like"],
            0.03,
            int(float(5 * 60000) / milliseconds_per_instance),
            2,
        )

        periodic_predictor_cols = {
            "acc_phone": ["acc_phone_x", "acc_phone_y", "acc_phone_z"],
            "acc_watch": ["acc_watch_x", "acc_watch_y", "acc_watch_z"],
            "gyr_phone": ["gyr_phone_x", "gyr_phone_y", "gyr_phone_z"],
            "gyr_watch": ["gyr_watch_x", "gyr_watch_y", "gyr_watch_z"],
            "mag_phone": ["mag_phone_x", "mag_phone_y", "mag_phone_z"],
            "mag_watch": ["mag_watch_x", "mag_watch_y", "mag_watch_z"],
        }

        print("computing frequency abstractions ...")
        dataset = FreqAbs.abstract_frequency(
            copy.deepcopy(dataset),
            periodic_predictor_cols,
            int(float(10000) / milliseconds_per_instance),
            fs,
        )

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1 - window_overlap) * ws)
        dataset = dataset.iloc[::skip_points, :]

        dataset.to_csv(DATA_PATH / RESULT_FNAME)
        print("saved dataset to", DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(
            dataset,
            [
                "acc_phone_x",
                "gyr_phone_x",
                "hr_watch_rate",
                "light_phone_lux",
                "mag_phone_x",
                "press_phone_",
                "pca_1",
                "label",
            ],
            ["like", "like", "like", "like", "like", "like", "like", "like"],
            ["line", "line", "line", "line", "line", "line", "line", "points"],
        )

        DataViz.plot_dataset(
            dataset,
            [
                "acc_phone_z_freq_energy",
                "acc_phone_sma",
                "acc_phone_z_temp_mad_ws_120",
                "light_phone_lux_temp_sum_ws_120",
                "label",
            ],
            ["like", "like", "like", "like", "like"],
            ["line", "line", "line", "line", "points"],
            # figsize=(10,12)
        )
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="final",
        help="Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ",
        choices=["aggregation", "frequency", "final"],
    )

    FLAGS, unparsed = parser.parse_known_args()

    main()
