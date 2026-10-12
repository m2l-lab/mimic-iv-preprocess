from __future__ import absolute_import
from __future__ import print_function
 
import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
 
from mimic3benchmark.scripts.binning_imputation import parse_timeseries, bin_timeseries, print_unparsable_summary
 
 
def is_target_ts_file(filename, suffix):
    if suffix == '_s':
        return "timeseries_s" in filename
    return "timeseries" in filename and "timeseries_s" not in filename
 
 
def process_partition(args, partition, unparsable_stats, eps=1e-6, n_hours=48):
    suffix = '_s' if args.variant == 'storetime' else ''
 
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
 
    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: is_target_ts_file(x, suffix), os.listdir(patient_folder)))
 
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
 
                # empty label file
                if label_df.shape[0] == 0:
                    continue
                icustay = label_df['Icustay'].iloc[0]
                
                mortality = int(label_df.iloc[0][args.mortality_column])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue
 
                if los < n_hours - eps:
                    continue
 
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
 
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]
 
                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue
 
                ts_df = parse_timeseries(header, ts_lines)
                binned_df = bin_timeseries(ts_df, window_hours=n_hours,
                                          include_mask=args.include_mask,
                                          stats=unparsable_stats)
 
                output_ts_filename = patient + "_" + ts_filename
                binned_df.to_csv(os.path.join(output_dir, output_ts_filename), index=False)
 
                xy_pairs.append((output_ts_filename, icustay, mortality))
 
    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)
 
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, icustay, y) in xy_pairs:
            listfile.write('{},0,{},{:d}\n'.format(x, icustay, y))
 
 
def main():
    parser = argparse.ArgumentParser(description="in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="path to root folder.")
    parser.add_argument('output_path', type=str, help="directory where data should be stored.")
    parser.add_argument('--variant', type=str, default='storetime', choices=['charttime', 'storetime'],
                        help="Which timestamp variant to process: 'charttime' or 'storetime' (default).")
    parser.add_argument('--mortality_column', type=str, default='Mortality',
                        choices=['Mortality', 'Mortality_1yr'],
                        help="Which label to use as target: 'Mortality' (default, "
                             "in-hospital) or 'Mortality_1yr' (1-year mortality).")
    parser.add_argument('--include_mask', dest='include_mask', action='store_true', default=True,
                        help="Add a binary missingness-mask per variable (default: on). "
                             "Use --no-include_mask for a pipeline without masks.")
    parser.add_argument('--no-include_mask', dest='include_mask', action='store_false')
    args, _ = parser.parse_known_args()
 
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
 
    unparsable_stats = {}
    process_partition(args, "test", unparsable_stats)
    process_partition(args, "train", unparsable_stats)
    print_unparsable_summary(unparsable_stats)
 
 
if __name__ == '__main__':
    main()