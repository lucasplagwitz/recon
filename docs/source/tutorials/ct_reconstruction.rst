.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_ct_reconstruction.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorials_ct_reconstruction.py:


02. Reconstruction I
================
This tutorial demonstrates the reconstruction of a
measurement obtained in computer tomography.
As mathematical construct the radon transformation is used here.
The implementations of skimage (radon, iradon) are used.

We create a scenario with a


.. code-block:: default

    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.data import shepp_logan_phantom
    from skimage.transform import rescale

    from recon.utils import psnr
    from recon.operator.ct_radon import CtRt
    from recon.interfaces import Recon, ReconBregman, Smoothing, SmoothBregman

    image = shepp_logan_phantom()
    image = rescale(image, scale=0.2, mode='reflect', multichannel=False)
    image = image / np.max(image)

    ny, nx = image.shape

    ntheta = 180
    theta = np.linspace(0., 180, ntheta, endpoint=False)

    sigma = 0.03

    R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)
    y = R*image.ravel()
    n = np.random.normal(0, sigma*np.max(y), size=y.shape)
    y = y + n

    x_rec = np.reshape(R.inv*y.ravel(), image.shape)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(np.reshape(y, R.image_dim), cmap='gray')
    axs[1].set_title('Data')
    axs[1].axis('tight')
    axs[2].imshow(x_rec, cmap='gray')
    axs[2].set_title("Reconstruction - PSNR: "+str(psnr(image, x_rec)))
    axs[2].axis('tight')
    fig.tight_layout()

    plt.show()




.. image:: /tutorials/images/sphx_glr_ct_reconstruction_001.png
    :alt: Model, Data, Reconstruction - PSNR: 26.42
    :class: sphx-glr-single-img





Similar to the Smoothing Tutorial a saddle point problem between
data fidelity and regularization is solved. For this purpose the Radon
operator is simply passed to the data term. The Recon interface takes care
of everything for the user.
We add a quick comparison to the solution that arises
when one should first reconstruct and then regularize.


.. code-block:: default


    rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tv', alpha=0.05, lam=1, tau='calc')
    x_tv = rec.solve(data=y.ravel(), max_iter=1000, tol=1e-5)

    smooth = Smoothing(domain_shape=image.shape, reg_mode='tv', alpha=0.05, lam=1, tau='calc')
    x_succession = smooth.solve(R.inv*y.ravel(), max_iter=1000, tol=1e-4)


    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(np.reshape(x_tv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
    axs[1].set_title('TV-Recon - PSNR:'+str(psnr(image, x_tv)))
    axs[1].axis('tight')
    axs[2].imshow(np.reshape(x_succession, R.domain_dim), vmin=0, vmax=1, cmap='gray')
    axs[2].set_title("Smooth $R^{-1}$*y - PSNR: "+str(psnr(image, x_succession)))
    axs[2].axis('tight')
    fig.tight_layout()

    plt.show()




.. image:: /tutorials/images/sphx_glr_ct_reconstruction_002.png
    :alt: Model, TV-Recon - PSNR:22.7, Smooth $R^{-1}$*y - PSNR: 25.48
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

     Early stopping.
     Early stopping.




Bregman versions


.. code-block:: default


    rec = ReconBregman(operator=R,
                       domain_shape=image.shape,
                       reg_mode='tv',
                       alpha=0.6,
                       lam=1,
                       assessment=sigma*np.max(y)*np.sqrt(np.prod(n.shape)),
                       tau='calc')
    breg_tv = rec.solve(data=y.ravel(), max_iter=1000, tol=1e-4)

    breg_smoothing = SmoothBregman(domain_shape=image.shape,
                                   reg_mode='tv',
                                   alpha=0.6,
                                   lam=1,
                                   tau='calc',
                                   plot_iteration=False,
                                   assessment=np.linalg.norm(R.inv*n.ravel(), 2))

    u_breg = breg_smoothing.solve(data=R.inv*y.ravel(), max_iter=1000, tol=1e-4)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(np.reshape(breg_tv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
    axs[1].set_title('TV-Breg - PSNR:'+str(psnr(image, breg_tv)))
    axs[1].axis('tight')
    axs[2].imshow(np.reshape(u_breg, R.domain_dim), vmin=0, vmax=1, cmap='gray')
    axs[2].set_title("Breg $R^{-1}$*y - PSNR: "+str(psnr(image, u_breg)))
    axs[2].axis('tight')
    fig.tight_layout()

    plt.show()




.. image:: /tutorials/images/sphx_glr_ct_reconstruction_003.png
    :alt: Model, TV-Breg - PSNR:28.74, Breg $R^{-1}$*y - PSNR: 27.1
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    current norm error: 1339.95748447
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 300.721536671
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 173.365584496
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 118.06560146
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 103.565528054
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 99.5171759621
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 96.9973524503
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 95.2163722957
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 93.9087984339
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 92.9020426723
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 92.0478141068
    runs till norm <: 91.5479594226
     Early stopping.
    current norm error: 16.7551090171
    runs till norm <: 2.28223009764
     Early stopping.
    1.1003151316
    current norm error: 9.49722051065
    runs till norm <: 2.28223009764
     Early stopping.
    1.69501166931
    current norm error: 5.60130635652
    runs till norm <: 2.28223009764
     Early stopping.
    2.1153413037
    current norm error: 3.41280723646
    runs till norm <: 2.28223009764
     Early stopping.
    2.47019462602
    current norm error: 2.81964425677
    runs till norm <: 2.28223009764
     Early stopping.
    2.69791611562
    current norm error: 2.5498024723
    runs till norm <: 2.28223009764
     Early stopping.
    2.66587723512
    current norm error: 2.36876071651
    runs till norm <: 2.28223009764
     Early stopping.




Conclusion
Further tests will follow. It seems that the L2 standard is not the best
choice for the radon-sinogram space.
... -> test L1-Norm


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 20 minutes  2.989 seconds)


.. _sphx_glr_download_tutorials_ct_reconstruction.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: ct_reconstruction.py <ct_reconstruction.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: ct_reconstruction.ipynb <ct_reconstruction.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
