# Image Inpainting for CSC420

The testing data is at the subfolder for each methods.

## To run the code:

### Patch-based

- To Run the Patch-Based method, execute `python patch/patched_base.py`
- The test data forPatch-Based method is in `patch/data`

### Diffusion-based

- To Run the Diffusion-Based method, execute `python diffusion/method3.py`
- The test data for Diffusion-Based method is in `diffusion/data`
- The result for Diffusion-Based method will be in `diffusion/save`

### Machine Learning

To run this model, you will need to install: PyTorch, torchvision, tensorboardX, pyyaml

- To train the Machine Learning model, execute `python machine_learning/train.py`
- To test the Machine Learning model, execute `python machine_learning/test_single.py --image machine_learning/examples/CelebA/4.jpg --mask machine_learning/examples/center_mask_256.png --output output.png`
- The examples for testing are at `machine_learning/examples`
