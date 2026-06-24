"""
Vertebra Body and Posterior Process Separation

This script separates vertebral bodies from posterior elements
in labeled 3D vertebra segmentation masks using PCA, Gaussian
Mixture Models, and connected component refinement.

Input:
    Labeled vertebra mask (.nii.gz)

Output:
    Vertebral body labels: N
    Posterior process labels: N + 1000
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, label
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import SimpleITK as sitk

def extract_body_and_processes(mask_3d, n_clusters=2):
    coords = np.array(np.nonzero(mask_3d)).T
    if len(coords) < 10:
        return mask_3d > 0, np.zeros_like(mask_3d, dtype=bool)

    com = np.array(center_of_mass(mask_3d))
    pca = PCA(n_components=3).fit(coords)

    long_axis = pca.components_[0]

    z_axis = np.array([0, 0, 1])
    cos_angle = np.abs(np.dot(long_axis, z_axis))
    if cos_angle > 0.9: 
        perp_axis = pca.components_[1]
    else:
        perp_axis = np.cross(long_axis, z_axis)
        perp_axis /= np.linalg.norm(perp_axis)

    projection_matrix = np.vstack([long_axis, perp_axis])

    proj_2d = np.dot(coords - com, projection_matrix.T)

    gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(proj_2d)
    labels = gmm.predict(proj_2d)

    cluster_means = [np.mean(proj_2d[labels == i], axis=0) for i in range(n_clusters)]
    dists_to_center = [np.linalg.norm(mean) for mean in cluster_means]
    body_cluster = np.argmin(dists_to_center)

    body_mask = np.zeros_like(mask_3d, dtype=bool)
    process_mask = np.zeros_like(mask_3d, dtype=bool)
    body_mask[tuple(coords[labels == body_cluster].T)] = True
    process_mask[tuple(coords[labels != body_cluster].T)] = True

    return body_mask, process_mask

def refine_labels_with_sagittal_connected_components(body_mask, process_mask):
    for x in range(process_mask.shape[0]):
        pe_slice = process_mask[x, :, :]
        if np.sum(pe_slice) == 0:
            continue

        pe_slice_sitk = sitk.GetImageFromArray(pe_slice.astype(np.uint8))
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled = cc_filter.Execute(pe_slice_sitk)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labeled)
        n_labels = stats.GetNumberOfLabels()

        if n_labels > 1:
            label_sizes = {l: stats.GetNumberOfPixels(l) for l in stats.GetLabels()}
            sorted_labels = sorted(label_sizes.items(), key=lambda x: x[1], reverse=True)
            largest_label = sorted_labels[0][0]

            labeled_np = sitk.GetArrayFromImage(labeled)
            for label_id, _ in sorted_labels[1:]:
                small_comp = (labeled_np == label_id)
                body_mask[x, :, :][small_comp] = True
                process_mask[x, :, :][small_comp] = False

    return body_mask, process_mask

def enforce_body_connectivity(body_mask, process_mask, min_size=500):
    labeled_body, n_labels = label(body_mask)
    sizes = [(l, np.sum(labeled_body == l)) for l in range(1, n_labels+1)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    if len(sizes) == 0:
        return body_mask, process_mask

    main_body_label = sizes[0][0]
    new_body_mask = (labeled_body == main_body_label)
    
    for label_id, size in sizes[1:]:
        if size < min_size:
            to_move = (labeled_body == label_id)
            new_body_mask[to_move] = False
            process_mask[to_move] = True
    
    return new_body_mask, process_mask

def process_labelled_mask_in_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                mask_path = os.path.join(dirpath, filename)
                img = nib.load(mask_path)
                img_ras = nib.as_closest_canonical(img)
                data = img_ras.get_fdata()
                affine = img_ras.affine
                labels = np.unique(data)
                labels = labels[labels != 0]
            

                all_vertebrae_mask = np.zeros(data.shape, dtype=np.uint16)

                for i, label_value in enumerate(labels, start=1):
                    vertebra_mask = (data == label_value).astype(np.uint8)
                    body_mask, process_mask = extract_body_and_processes(vertebra_mask)
                    body_mask, process_mask = refine_labels_with_sagittal_connected_components(body_mask, process_mask)
                    body_mask, process_mask = enforce_body_connectivity(body_mask, process_mask)

                    all_vertebrae_mask[body_mask] = i
                    all_vertebrae_mask[process_mask] = i + 1000

                output_path = os.path.join(dirpath, "combined_vertebrae_mask_with_body_and_processes.nii.gz")
                nib.save(nib.Nifti1Image(all_vertebrae_mask, affine), output_path)
                print(f" Done: {output_path}")

if __name__ == "__main__":
    root_folder = input("Enter folder path: ")
    process_labelled_mask_in_folder(root_folder)
    



