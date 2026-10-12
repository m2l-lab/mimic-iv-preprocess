from io import StringIO
 
import numpy as np
import pandas as pd
 
 
CATEGORICAL_VARIABLES = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]
 
 
def parse_timeseries(header_line, ts_lines):
    csv_text = header_line + ''.join(ts_lines)
    return pd.read_csv(StringIO(csv_text))
 
 
def bin_timeseries(df, window_hours, categorical_vars=None, include_mask=True, eps=1e-6, stats=None):
    if categorical_vars is None:
        categorical_vars = CATEGORICAL_VARIABLES
 
    n_bins = max(1, int(np.ceil(window_hours - eps)))
    variables = [c for c in df.columns if c != 'Hours']
 
    work = df.copy()
    work['__bin__'] = np.floor(work['Hours'] + eps).astype(int)
    work['__bin__'] = work['__bin__'].clip(lower=0, upper=n_bins - 1)
 
    out = pd.DataFrame({'Hours': np.arange(n_bins)})
    mask_cols = {}
 
    for var in variables:
        sub = work[['__bin__', var]].dropna(subset=[var])
 
        if var in categorical_vars:
            agg = sub.groupby('__bin__')[var].agg(lambda x: x.mode().iloc[0])
        else:
            numeric_vals = pd.to_numeric(sub[var], errors='coerce')
            n_unparsable = int(numeric_vals.isnull().sum())
            if n_unparsable > 0 and stats is not None:
                bad_values = set(sub.loc[numeric_vals.isnull(), var].astype(str))
                entry = stats.setdefault(var, {'count': 0, 'examples': set()})
                entry['count'] += n_unparsable
                entry['examples'].update(bad_values)

            sub = sub.assign(**{var: numeric_vals})
            agg = sub.groupby('__bin__')[var].mean()
 
        col = agg.reindex(range(n_bins))
 
        if include_mask:
            mask_cols[var] = col.notnull().astype(int).values
        out[var] = col.values
 
    if include_mask:
        for var in variables:
            out['{}_mask'.format(var)] = mask_cols[var]
 
    return out
 
 
def print_unparsable_summary(stats, max_examples=5):
    if not stats:
        return
    print("\nSummary: non-numeric values found and treated as missing (NaN):")
    for var in sorted(stats.keys()):
        entry = stats[var]
        examples = sorted(entry['examples'])[:max_examples]
        more = "" if len(entry['examples']) <= max_examples else ", ..."
        print("  {}: {} value(s), e.g. {}{}".format(var, entry['count'], examples, more))
 