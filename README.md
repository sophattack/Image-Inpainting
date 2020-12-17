# Image Inpainting for CSC420

See [./data dir](./data/README.md) for notes on maintaining data



Notes on Patched Based Method:
- Patch-based methods are based on techniques that fill in the missing region patch-by-patch by searching for well-matching replacement patches (i.e., candidate patches) in the undamaged part of the image and copying them to corresponding locations. 
- In our proposal:
We will implement Ružić’s patch-based approach[3] because compared to other patch-based techniques, this algorithm makes use of the Markov Random Field to improve the computation efficiency.
- Ružić’s paper: Context-aware patch-based image inpainting using Markov random field modelling. 
- Context-aware patch selection can be used w/ greedy, multiple candidates, and global
- MRF is for global? If can't figure this out.. maybe use the context aware patch selection with an easier method?