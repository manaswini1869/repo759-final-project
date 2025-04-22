Lack of spatial Symmetry

- Language models have the CNNs (translational invariance? ) but lack vision transformer
- sin function the transformer wave fades as it goes away from the origin
- function images in the spatial domain center concentrated but faded around the edges
- FFT captures the primary frequencies and orientations of the sin components, showing symmetric bright spots at the center. these spots corresponds to sin waves while smoother, spread-out regions indicated the {Gaussian decay}(?)
- spatial to frequency domain concentrated spots
- offers a way to enhance the model's perception of complex patterns, shapes, and textures(?) by providing explicit frequency-based features
- helps in enhancing shape and symmetry recognition, edge detection and texture detail, generalization across domain with varied patterns, noise filtering and detail enhancement
- allows a reduction in the amount of spatial information needed by summarizing an image's structure in the frequency domain => can make prompts more parameter-efficient, especially useful in peft methods
- direct to a meaningful structures, making it easier for the mode to focus on relevant details without needing extensive fine-tuning
- give a clearer breakdown of which shapes and patterns model is focusing on,
- cross path symmetry
- using symmetrical features models will learn the patterns easier
- textures, remote sensing, medical imaging, material analysis
- fewer parameter concentration,
- including FFT-derived symmetry information in prompts enhance sensitivity to symmetrical features
- FFT provides enhances images segmentation tasks by providing freq domain features thats highlight the repetitive patterns
- leverage both spatial and frequency domain

https://arxiv.org/abs/2405.03003

https://github.com/FFTW/fftw3/blob/master/api/fftw3.h


- which treats the weight change ∆W as a matrix in the
spatial domain, and learns its sparse spectral coefficients.
Specifically, we first randomly select n spectral entries that
are shared across all layers. For each layer, FourierFT learns
n spectral coefficients located at these n selected entries and
then directly applies inverse discrete Fourier transform to
compute the updated ∆W. Therefore, fine-tuning a pretrained model with Lt layers only requires storing 2n entry
parameters and nLt coefficient parameters for FourierFT.

- Instead of storing full ΔW matrices for each layer, store only a few Fourier domain coefficients and reconstruct ΔW using inverse FFT.
- FourierFT selects only a few (say, C) frequency positions and learns (?)coefficients at those points. The rest are zero.
- // nvcc fourier.cu -lcufft -o fourier

https://docs.nvidia.com/cuda/cufftdx/examples.html