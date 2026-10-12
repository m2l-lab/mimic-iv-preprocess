"""Extract fixed-size CXR image embeddings using torchxrayvision's
DenseNet121 (CheXpert-pretrained weights), for the matched-subset stays
produced by cxr_matching.py.
"""

import os

import numpy as np
import pandas as pd


def build_image_path(image_dir, dicom_id, extension="jpg"):
    return os.path.join(image_dir, "{}.{}".format(dicom_id, extension))


def check_images_exist(matched_df, image_dir, extension="jpg"):
    paths = matched_df['dicom_id'].apply(lambda d: build_image_path(image_dir, d, extension))
    exists = paths.apply(os.path.exists)
    found_df = matched_df.loc[exists].copy()
    missing_df = matched_df.loc[~exists].copy()
    return found_df, missing_df


def _load_and_preprocess_image(path):
    import skimage.io
    import torchxrayvision as xrv
    import torchvision

    img = skimage.io.imread(path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]  # (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)
    return img  # (1, 224, 224) float array


def extract_image_embeddings(matched_df, image_dir, weights="densenet121-res224-chex",
                              extension="jpg", batch_size=16, device=None, include_predictions=False):
    """
    include_predictions : If True (default: False), also extract the 18 pathology-probability
        outputs alongside the 1024 dense columns.
    """
    import torch
    import torchxrayvision as xrv

    if len(matched_df) == 0:
        raise ValueError(
            "extract_image_embeddings() got 0 rows to process")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device), flush=True)
    model = xrv.models.DenseNet(weights=weights)
    model = model.to(device)
    model.eval()
    print("Model loaded.", flush=True)
    all_embeddings = []
    all_predictions = [] if include_predictions else None
    all_stay_ids = []

    rows = matched_df.to_dict("records")
    n_batches = (len(rows) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(rows), batch_size)):
        batch = rows[i:i + batch_size]
        imgs = []
        for r in batch:
            path = build_image_path(image_dir, r['dicom_id'], extension)
            imgs.append(_load_and_preprocess_image(path))
        batch_tensor = torch.from_numpy(np.stack(imgs, axis=0)).float().to(device)

        with torch.no_grad():
            emb = model.features(batch_tensor)
            if emb.dim() > 2:
                emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).flatten(1)
            emb = emb.cpu().numpy()

            if include_predictions:
                pred = model(batch_tensor).cpu().numpy()
                all_predictions.append(pred)

        all_embeddings.append(emb)
        all_stay_ids.extend([r['stay_id'] for r in batch])

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print("Batch {}/{} done ({} images so far)".format(
                batch_idx + 1, n_batches, len(all_stay_ids)), flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    cols = ["img_emb_{}".format(j) for j in range(embeddings.shape[1])]
    out = pd.DataFrame(embeddings, columns=cols)
    out.insert(0, "stay_id", all_stay_ids)

    if include_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        pred_cols = ["img_pred_{}".format(p) for p in model.pathologies]
        pred_df = pd.DataFrame(predictions, columns=pred_cols)
        out = pd.concat([out, pred_df], axis=1)

    return out


def main():
    import argparse
    from .cxr_matching import match_images_to_stays

    parser = argparse.ArgumentParser(
        description="Precompute CXR image embeddings for every stay-matched image.")
    parser.add_argument('--cxr_metadata', type=str, required=True,
                         help="Path to mimic-cxr-2.0.0-metadata.csv")
    parser.add_argument('--all_stays', type=str, required=True,
                         help="Path to root_path/all_stays.csv")
    parser.add_argument('--image_dir', type=str, required=True,
                         help="Flat directory of resized images, named <dicom_id>.<extension>")
    parser.add_argument('--extension', type=str, default='jpg')
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--weights', type=str, default='densenet121-res224-chex',
                         choices=['densenet121-res224-chex', 'densenet121-res224-mimic_ch', 'densenet121-res224-all'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--include_predictions', dest='include_predictions', action='store_true', default=False)
    parser.add_argument('--no-include_predictions', dest='include_predictions', action='store_false')
    args = parser.parse_args()
    print("Args: {}".format(vars(args)), flush=True)

    print("Matching CXR images to ICU stays (48h window, AP view, latest image)...", flush=True)
    matched = match_images_to_stays(args.cxr_metadata, args.all_stays)
    print("Stays with a qualifying image: {}".format(matched['stay_id'].nunique()), flush=True)
    if len(matched) == 0:
        raise RuntimeError(
            "0 images matched any stay. Check --cxr_metadata and --all_stays: verify column "
            "names (subject_id, stay_id, intime, outtime in all_stays.csv; subject_id, study_id, "
            "dicom_id, ViewPosition, StudyDate, StudyTime in cxr_metadata) and that date ranges "
            "actually overlap.")

    found, missing = check_images_exist(matched, args.image_dir, args.extension)
    print("Images found on disk: {} / {}".format(len(found), len(matched)), flush=True)
    if len(missing) > 0:
        print("Warning: {} matched image(s) not found on disk under {} and will be skipped, e.g. {}".format(
            len(missing), args.image_dir, missing['dicom_id'].head(3).tolist()), flush=True)
    if len(found) == 0:
        raise RuntimeError(
            "0 matched images actually exist under --image_dir='{}'. Check the path and that "
            "filenames follow <dicom_id>.<extension> (extension='{}').".format(
                args.image_dir, args.extension))

    print("Extracting embeddings for {} images (weights={})...".format(len(found), args.weights), flush=True)
    embeddings_df = extract_image_embeddings(found, args.image_dir, weights=args.weights,
                                              extension=args.extension, batch_size=args.batch_size,
                                              include_predictions=args.include_predictions)
    embeddings_df.to_csv(args.output_csv, index=False)
    n_emb_cols = sum(1 for c in embeddings_df.columns if c.startswith('img_emb_'))
    n_pred_cols = sum(1 for c in embeddings_df.columns if c.startswith('img_pred_'))
    print("Saved {} embeddings ({} dense dims{}) to {}".format(
        len(embeddings_df), n_emb_cols,
        " + {} prediction dims".format(n_pred_cols) if n_pred_cols else "",
        args.output_csv), flush=True)


if __name__ == '__main__':
    main()
