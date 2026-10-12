from __future__ import absolute_import
from __future__ import print_function
 
import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)
from tqdm import tqdm
 
from mimic3benchmark.scripts.binning_imputation import parse_timeseries, bin_timeseries, print_unparsable_summary
 
 
def is_target_ts_file(filename, suffix):
    if suffix == '_s':
        return "timeseries_s" in filename
    return "timeseries" in filename and "timeseries_s" not in filename
 
 
def process_partition(args, definitions, code_to_group, id_to_group, group_to_id,
                      partition, unparsable_stats, eps=1e-6, n_hours=48):
    suffix = '_s' if args.variant == 'storetime' else ''
 
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
 
    xty_triples = []
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
 
                los = 24.0 * label_df.iloc[0]['Length of Stay']
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
 
                cur_labels = [0 for i in range(len(id_to_group))]
 
                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"icd_code": str})
                diagnoses_df = diagnoses_df[diagnoses_df.stay_id == icustay]
                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['icd_code']
                        if code in code_to_group:
                            group = code_to_group[code]
                            group_id = group_to_id[group]
                            cur_labels[group_id] = 1
                        else:
                            print(f'{code} code not found')    
                # import pdb; pdb.set_trace()
                cur_labels = [x for (i, x) in enumerate(cur_labels)
                              if definitions[id_to_group[i]]['use_in_benchmark']]
 
                xty_triples.append((output_ts_filename, n_hours, icustay, cur_labels))
    
 
    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)
 
    codes_in_benchmark = [x for x in id_to_group
                          if definitions[x]['use_in_benchmark']]
 
    listfile_header = "stay,period_length,stay_id," + ",".join(codes_in_benchmark)
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, t, stay_id, y) in xty_triples:
            labels = ','.join(map(str, y))
            listfile.write('{},{:.6f},{},{}\n'.format(x, t, stay_id, labels))
 
 
def main():
    parser = argparse.ArgumentParser(description="phenotype classification task.")
    parser.add_argument('root_path', type=str, help="path to root folder.")
    parser.add_argument('output_path', type=str, help="directory where data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                        help='YAML file with phenotype definitions.')
    parser.add_argument('--variant', type=str, default='storetime', choices=['charttime', 'storetime'],
                        help="Which timestamp variant to process: 'charttime' or 'storetime' (default).")
    parser.add_argument('--include_mask', dest='include_mask', action='store_true', default=True,
                        help="Add a binary missingness-mask per variable (default: on). "
                             "Use --no-include_mask for a pipeline without masks.")
    parser.add_argument('--no-include_mask', dest='include_mask', action='store_false')
    args, _ = parser.parse_known_args()
 
    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.safe_load(definitions_file)
 
    code_to_group = {}
 
   
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group
            else:
                print(f'code, {code}')
                assert code_to_group[code] == group
 
    # import pdb;pdb.set_trace()
    # ['Diabetes mellitus with complication', ]
    # 'ICD-10-CM CODE' 'Default CCSR CATEGORY DESCRIPTION IP'
    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))
 
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
 
    unparsable_stats = {}
    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "test", unparsable_stats)
    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "train", unparsable_stats)
    print_unparsable_summary(unparsable_stats)
 
 
if __name__ == '__main__':
    main()
 