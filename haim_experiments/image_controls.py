"""Controls for the dimensionality of the image block: PCA, Gaussian noise, permuted rows."""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd


IMAGE_TRANSFORMS = ('none', 'pca', 'gaussian_noise', 'permuted')


def transform_image_block(image_df, split_stay_ids, transform='none', pca_dim=20,
                          noise_dim=None, seed=0):
    if transform not in IMAGE_TRANSFORMS:
        raise ValueError("Unknown image transform '{}'".format(transform))
    feat_cols = [c for c in image_df.columns if c != 'stay_id']
    info = {'image_transform': transform, 'n_image_columns_in': len(feat_cols)}
    if transform == 'none':
        info['n_image_columns_out'] = len(feat_cols)
        return image_df, info

    df = image_df.drop_duplicates('stay_id').reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if transform == 'pca':
        from sklearn.decomposition import PCA
        train_mask = df.stay_id.isin(set(split_stay_ids['train'])).values
        if train_mask.sum() < pca_dim:
            raise ValueError('Not enough train stays with images ({}) for {} PCA components'.format(
                int(train_mask.sum()), pca_dim))
        X = df[feat_cols].values.astype(float)
        pca = PCA(n_components=pca_dim, random_state=seed).fit(X[train_mask])
        out = pd.DataFrame(pca.transform(X), columns=['img_pca_{}'.format(i) for i in range(pca_dim)])
        info['pca_dim'] = pca_dim
        info['pca_explained_variance_ratio_total'] = float(pca.explained_variance_ratio_.sum())
        info['pca_explained_variance_ratio'] = [float(v) for v in pca.explained_variance_ratio_]
        print("Image PCA: {} -> {} components (fit on {} train stays), explained variance {:.1%}".format(
            len(feat_cols), pca_dim, int(train_mask.sum()), info['pca_explained_variance_ratio_total']))

    elif transform == 'gaussian_noise':
        d = noise_dim or len(feat_cols)
        out = pd.DataFrame(rng.standard_normal((len(df), d)),
                           columns=['img_noise_{}'.format(i) for i in range(d)])
        info['noise_dim'] = d
        print("Image block replaced by {} iid N(0,1) noise columns".format(d))

    else:  # permuted within each split
        out = df[feat_cols].copy()
        n_permuted = 0
        for ids in split_stay_ids.values():
            idx = np.where(df.stay_id.isin(set(ids)).values)[0]
            out.iloc[idx] = out.iloc[rng.permutation(idx)].values
            n_permuted += len(idx)
        info['n_stays_permuted'] = n_permuted
        print("Image rows permuted across stays within each split ({} stays)".format(n_permuted))

    out.insert(0, 'stay_id', df.stay_id.values)
    info['n_image_columns_out'] = out.shape[1] - 1
    info['image_control_seed'] = seed
    return out, info
