# Primary-source availability audit

Checked **2026-11-10**. Findings describe resources inspected at that time, not a promise that external links remain reachable. This package includes original code and documentation; copyrighted paper PDFs and official project imagery are not redistributed.

## RadSimReal publication and repository

| Resource | Checked content | Outcome |
|---|---|---|
| [CVPR 2024 official entry](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html) | Publication identity and paper/supplement links | CVF direct fetch returned HTTP 403 in the available browser tool; use accessible author manuscript below. |
| [arXiv author manuscript](https://arxiv.org/html/2404.18150v1) | Entire main paper, references and supplementary Appendices A–D | Read. The supplement is included here, with PSF/noise measurement, five operator equations, complexity analysis and RADDet scene split. No byte-for-byte comparison against the CVF PDFs was performed. |
| [arXiv record](https://arxiv.org/abs/2404.18150) | Submission metadata | Identifies Bialer and Haitman's manuscript. |
| [Official project site](https://yuvalhg.github.io/RadSimReal/) | Method and result descriptions and linked resources | Read; qualitative and detector results are presented, not executable simulator instructions. |
| [Official GitHub repository](https://github.com/yuvalHG/RadSimReal) | Repository tree and README | Audited commit `91a024e9fcb31c8555b9ee3140acd534865ec0d4`. README and website assets/docs were available; executable author simulator/training source, released weights, measured PSF arrays, author synthetic dataset and configuration files were not present in the inspected tree. |

The central source-to-code contract is Appendix B.2 Eq. (4)/(5), weighted shifted-PSF addition/convolution. Sensor calibration follows Appendix A's procedure conceptually; numeric calibration files are an external input. Exactness is conditional on representation and sampling. See [REPRODUCTION_NOTES.md](../REPRODUCTION_NOTES.md) for implementation judgments and limitations.

## Original downstream detector references

These references do not determine the unpublished RadSimReal adaptations.

| Reference | Primary source | Relevant finding and consequence |
|---|---|---|
| Coordinate-augmented U-Net | Zhang, Li, Wenger, [FMCW FCN paper](https://arxiv.org/pdf/1902.05394) | Original reference predicts presence and camera-coordinate maps from normalized complex range-Doppler inputs. Its published task does not directly supply RadSimReal's RA box heads. The package's coordinate-U-Net detector is a disclosed adaptation. |
| RADDet | Zhang et al., [official repository](https://github.com/ZhangAoCanada/RADDet), [paper record](https://arxiv.org/abs/2105.00363) and [configuration](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/master/config.json) | Separate TensorFlow source, datasets, configuration, anchors and advertised checkpoint links exist. V2 includes a PyTorch port of pinned released source, with adaptations recorded in the detector implementation guide. These resources do not supply RadSimReal's experiment-specific weights or scene-disjoint results. |
| Probabilistic | Dong et al., [primary manuscript](https://arxiv.org/html/2004.05310v2) | Describes an oriented anchor detector with uncertainty-aware regression. V2 implements the described architecture/equations with disclosed reconstruction choices. Exact RadSimReal transplantation remains unspecified; public RADDet/CARRADA labels do not supply genuine car yaw targets. |

No RadSimReal-specific optimizer, learning-rate schedule, epoch count, augmentation recipe, box-head specification or exact coordinate-channel encoding was found in the inspected publication/repository. Settings from a cited detector's original paper cannot be promoted to confirmed RadSimReal settings. This project's executable baseline chooses and records its own settings.

RADDet's original [requirements file](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/master/requirements.txt) was also inspected; it pins historical TensorFlow/NumPy and supporting packages. They are recorded in the external-data guide, not silently mixed into this project's PyTorch environment. The external `train.py` uses TensorFlow/Keras; no external checkpoint download or execution of that repository was performed here.

## External data and calibration

- [RADDet official repository](https://github.com/ZhangAoCanada/RADDet) supplies dataset access routes and describes original tensor/annotation formats. Its data access notes state that former Google Drive links became unavailable; alternative links are provided. Successful downloading of the full dataset or checkpoint was not established in this implementation audit.
- [CARRADA primary dataset page](https://arthurouaknine.github.io/codeanddata/carrada) links data and [official source](https://github.com/valeoai/carrada_dataset). CARRADA data and code have their own licenses; this project's license does not override them.
- [CRUW primary site](https://www.cruwdataset.org/home), [resources](https://www.cruwdataset.org/resources), and [introduction](https://www.cruwdataset.org/introduction) describe the radar/camera dataset and annotation semantics. Their native radar annotations provide class/center information, which is insufficient to reconstruct the exact unpublished CFAR-derived boxes used in RadSimReal.
- Author PSFs, noise arrays, RF material parameters, CARLA scene generation assets, CARRADA/CRUW split manifests and derived CRUW boxes were not found among inspected RadSimReal resources.

The absence statements concern the inspected sources and commit. They do not assert that the authors never shared additional materials privately or that no third-party implementation exists.

## Bounding-box domain audit

The [RadSimReal PDF](https://arxiv.org/pdf/2404.18150), Fig. 5 (manuscript page 8), draws predicted/ground-truth boxes on axes labeled Cartesian x/y in metres. Section 4 describes AP and IoU thresholds without providing the exact metric code or a complete box-domain conversion specification. The figure is direct evidence of the displayed coordinate system, not by itself a formal specification of every evaluation operation.

The cited [RADDet primary paper](https://arxiv.org/html/2105.00363v1), Sec. IV-C2, explicitly has a Cartesian 2D head after polar-to-Cartesian feature transformation. The cited [Dong probabilistic detector](https://arxiv.org/html/2004.05310v2) explicitly predicts oriented boxes in BEV space. Taken together, these sources strongly indicate Cartesian/BEV evaluation conventions in the published comparison, but the exact RadSimReal adaptations are still unavailable.

The retained **legacy v1** detector evaluates normalized RA-image IoU. V2 provides physical Cartesian axis-aligned IoU/AP and optional oriented polygon IoU/AP when genuine yaw labels exist. RADDet's original Cartesian rectangles are preserved through a shared invertible affine, which preserves their IoU. CARRADA uses an explicitly named Cartesian enclosure of native radar annotation support; that conversion is not asserted to equal the unpublished RadSimReal conversion. Matching thresholds alone do not establish evaluator parity. Recovering an original physical footprint or yaw from an RA enclosing rectangle remains impossible in general.

## Current v2 implementation status

V2 adds native RADDet/CARRADA readers, explicit source-derived sensor profiles, train-only PSF/noise measurement, mesh and moving-OBJ reflectors, native-format synthetic exports and all three downstream model families. Public-source anchors and implementation adaptations are recorded in [DETECTOR_IMPLEMENTATION.md](DETECTOR_IMPLEMENTATION.md), [NATIVE_FORMATS.md](NATIVE_FORMATS.md) and [SCATTERING.md](SCATTERING.md). [HPC_GUIDE.md](HPC_GUIDE.md) describes execution beside the user's datasets. These additions resolve the former RA-only engineering limitation; they do not supply unavailable author assets, settings or calibration measurements. Fixture execution and actual-recording validation are tracked separately in the current test report.

## Interpretation rules

1. Published processing stages and equation identities are source facts.
2. This project's domain conventions, geometry, scattering choices, noise law, learning recipe and coordinate encoding are engineering decisions unless explicitly tied to an available author artifact.
3. Small synthetic smoke tests establish functioning software. Only a controlled real-data study with known calibration and held-out scenes can assess the sim-to-real claims.
4. Published accuracy, FID and speedup numbers are reference results, not measurements obtained by this package.
