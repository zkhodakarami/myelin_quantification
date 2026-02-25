"""
custom_trainer.py
=================
Custom nnU-Net v2 trainer with histology-specific data augmentations for
white-matter mask segmentation.

Extends ``nnUNetTrainerDA5`` (the most aggressive built-in augmentation
variant) with:

* Full 180-degree rotation (histology slides have no canonical orientation)
* Wider scaling range  (0.5 -- 1.6)
* Higher probability for spatial transforms
* Per-channel additive colour shifts to simulate H&E stain variability
* Extra Gaussian noise band for acquisition-noise robustness

Install into nnU-Net
--------------------
Run ``install_trainer.py`` **once** after installing nnunetv2:

    python install_trainer.py

Then train with:

    nnUNetv2_train DATASET_ID 2d FOLD -tr nnUNetTrainerHistoAug
"""

from typing import List, Tuple, Union

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    MedianFilterTransform,
    SharpeningTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
    SpatialTransform,
    TransposeAxesTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar

from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import (
    ApplyRandomBinaryOperatorTransform,
    MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)
from nnunetv2.training.data_augmentation.custom_transforms.masking import (
    MaskTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import (
    ConvertSegmentationToRegionsTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)


# ---------------------------------------------------------------------------
# Helper functions (same pattern as DA5)
# ---------------------------------------------------------------------------


def _brightnessadditive_localgamma_transform_scale(x, y):
    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))


def _brightness_gradient_additive_max_strength(_x, _y):
    return (
        np.random.uniform(-5, -1)
        if np.random.uniform() < 0.5
        else np.random.uniform(1, 5)
    )


def _local_gamma_gamma():
    return (
        np.random.uniform(0.01, 0.8)
        if np.random.uniform() < 0.5
        else np.random.uniform(1.5, 4)
    )


# ---------------------------------------------------------------------------
# Custom colour-shift transform for histology stain variability
# ---------------------------------------------------------------------------


class HistoColorJitterTransform(AbstractTransform):
    """Per-channel additive + multiplicative colour perturbation.

    Mimics stain-intensity variability commonly seen in H&E-stained
    histology images.  Applied independently per channel.
    """

    def __init__(
        self,
        additive_range: tuple = (-0.1, 0.1),
        multiplicative_range: tuple = (0.85, 1.15),
        p_per_sample: float = 0.3,
        p_per_channel: float = 0.5,
        data_key: str = "data",
    ):
        self.additive_range = additive_range
        self.multiplicative_range = multiplicative_range
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data.shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        add = np.random.uniform(*self.additive_range)
                        mult = np.random.uniform(*self.multiplicative_range)
                        data[b, c] = data[b, c] * mult + add
        data_dict[self.data_key] = data
        return data_dict


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class nnUNetTrainerHistoAug(nnUNetTrainerDA5):
    """nnU-Net trainer with histology-optimised augmentations.

    Key differences from ``nnUNetTrainerDA5``:

    * Always uses full 180-degree rotation (histology has no canonical
      orientation).
    * Wider scale range (0.5 -- 1.6 vs 0.7 -- 1.43).
    * Higher rotation/scale probability (0.6 / 0.3 vs 0.4 / 0.2).
    * ``HistoColorJitterTransform`` for stain variability.
    * Increased Gaussian noise probability (0.15 vs 0.1).
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        if dim == 2:
            do_dummy_2d_data_aug = False
            rotation_for_DA = (-np.pi, np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            from nnunetv2.configuration import ANISO_THRESHOLD

            do_dummy_2d_data_aug = (
                max(patch_size) / patch_size[0]
            ) > ANISO_THRESHOLD
            rotation_for_DA = (-np.pi, np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError(f"Unsupported dimensionality: {dim}")

        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.5, 1.6),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(
            f"[HistoAug] do_dummy_2d_data_aug: {do_dummy_2d_data_aug}"
        )
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        matching_axes = np.array(
            [sum([i == j for j in patch_size]) for i in patch_size]
        )
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms: list = []
        tr_transforms.append(RenameTransform("target", "seg", True))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # --- Spatial transforms (wider scale, higher probability) ----------
        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA,
                angle_y=rotation_for_DA,
                angle_z=rotation_for_DA,
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.5, 1.6),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.3,
                p_rot_per_sample=0.6,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3),
                    axes=valid_axes,
                    data_key="data",
                    label_key="seg",
                    p_per_sample=0.5,
                ),
            )
        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(
                    valid_axes,
                    data_key="data",
                    label_key="seg",
                    p_per_sample=0.5,
                )
            )

        # --- Blur / noise --------------------------------------------------
        tr_transforms.append(
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )
        tr_transforms.append(
            GaussianBlurTransform(
                (0.3, 1.5),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))

        # --- Colour / intensity (histology-specific) -----------------------
        tr_transforms.append(
            HistoColorJitterTransform(
                additive_range=(-0.1, 0.1),
                multiplicative_range=(0.85, 1.15),
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            BrightnessTransform(
                0,
                0.5,
                per_channel=True,
                p_per_sample=0.15,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            ContrastAugmentationTransform(
                contrast_range=(0.5, 2),
                preserve_range=True,
                per_channel=True,
                data_key="data",
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.25, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.15,
                ignore_axes=ignore_axes,
            )
        )

        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )
        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=False,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.3,
            )
        )

        # --- Mirror --------------------------------------------------------
        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        # --- Cutout / blank rectangles -------------------------------------
        tr_transforms.append(
            BlankRectangleTransform(
                [[max(1, p // 10), p // 3] for p in patch_size],
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5,
            )
        )

        # --- Local intensity transforms ------------------------------------
        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        # --- Mask / label housekeeping -------------------------------------
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError(
                    "ignore label not yet supported in cascade"
                )
            assert (
                foreground_labels is not None
            ), "We need all_labels for cascade augmentations"
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(
                MoveSegAsOneHotToData(1, use_labels, "seg", "data")
            )
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1,
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                )
            )

        tr_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    (
                        list(regions) + [ignore_label]
                        if ignore_label is not None
                        else regions
                    ),
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales,
                    0,
                    input_key="target",
                    output_key="target",
                )
            )

        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms
