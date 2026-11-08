# Sources and third-party notices

This repository contains PyTorch implementations of the methods used in the CABLE experiments. Baselines are adaptations to the common environment and training pipeline, not verbatim executions of the original repositories.

- ClonEx-SAC and the Continual World PackNet adaptation: https://github.com/awarelab/continual_world and the code supplement to *Disentangling Transfer in Continual Reinforcement Learning* (NeurIPS 2022): https://papers.neurips.cc/paper_files/paper/2022/file/2938ad0434a6506b125d8adaff084a4a-Supplemental-Conference.zip
- RECALL: https://github.com/Sweety-dm/RECALL
- Meta-World: https://github.com/Farama-Foundation/Metaworld
- PyTorch: https://github.com/pytorch/pytorch
- MuJoCo: https://github.com/google-deepmind/mujoco

The release omits downloaded third-party source trees, papers, and assets. Dependencies are installed separately and remain subject to their own licenses. Algorithmic sources are acknowledged above; original CABLE project code is covered by the root license.

ClonEx-SAC uses teacher-to-current Gaussian KL and gradient averaging in this implementation. RECALL uses current-to-historical policy KL, historical RL replay, PopArt, and no historical critic-distillation penalty. PackNet uses actor-backbone allocation and 100,000 replay-based retraining updates after each nonfinal task; fine-tuning uses a single-head SAC network. Consult the archived configurations for the actual run settings.
