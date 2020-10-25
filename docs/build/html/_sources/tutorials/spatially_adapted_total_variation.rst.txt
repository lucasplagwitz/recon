.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_spatially_adapted_total_variation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorials_spatially_adapted_total_variation.py:


05. Spatially Adapted Total Variation
================

Here a locally adapted regularization is shown.
For this purpose the SATV algorithm was implemented.
The application and the nurzen are shown.
Furthermore, TV is compared with TGV in the context of the local regularization.


.. code-block:: default

    import numpy as np
    import matplotlib.pyplot as plt
    from recon.utils.utils import psnr
    from recon.utils.images import local_tss
    from recon.interfaces import SATV, Smoothing

    image = local_tss()

    noise_sigma = 0.2*np.max(image)

    noisy_image = image + np.random.normal(0, noise_sigma, size=image.shape)

    # TV smoothing small alpha
    tv_smoothing = Smoothing(domain_shape=image.shape, reg_mode='tv', alpha=0.3, lam=1)
    u_tv = tv_smoothing.solve(data=noisy_image, max_iter=5000, tol=1e-4)


    f = plt.figure(figsize=(6, 3))
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.gray()
    plt.imshow(image, vmin=0, vmax=np.max(image))
    plt.title("GT - PSNR: "+str(psnr(image, image)))
    f.add_subplot(1, 2, 2)
    plt.imshow(u_tv, vmin=0, vmax=np.max(image))
    plt.title("TV - PSNR: "+str(psnr(image, u_tv)))
    plt.axis('off')
    plt.gray()
    plt.show(block=False)




.. image:: /tutorials/images/sphx_glr_spatially_adapted_total_variation_001.png
    :alt: GT - PSNR: -1, TV - PSNR: 20.07
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

     Early stopping.




...


.. code-block:: default


    satv_obj = SATV(domain_shape=image.shape,
                    reg_mode='tv',
                    lam=0.3,
                    alpha=0.3,
                    plot_iteration=False,
                    noise_sigma=noise_sigma,
                    assessment=noise_sigma*np.sqrt(np.prod(image.shape)))
    satv_solution = satv_obj.solve(noisy_image, max_iter=5000, tol=1e-4)

    f = plt.figure(figsize=(9, 3))
    f.add_subplot(1, 3, 1)
    plt.gray()
    plt.axis('off')
    plt.imshow(noisy_image, vmin=0, vmax=np.max(image))
    plt.title("Noisy - PSNR: "+str(psnr(image, noisy_image)))
    f.add_subplot(1, 3, 2)
    plt.gray()
    plt.imshow(satv_solution, vmin=0, vmax=np.max(image))
    plt.title("SATV - PSNR: "+str(psnr(image, satv_solution)))
    plt.axis('off')
    f.add_subplot(1, 3, 3)
    plt.gray()
    plt.imshow(np.reshape(satv_obj.lam, image.shape))
    plt.title("SATV-weight $\lambda$")
    plt.axis('off')
    plt.show()





.. image:: /tutorials/images/sphx_glr_spatially_adapted_total_variation_002.png
    :alt: Noisy - PSNR: 13.99, SATV - PSNR: 23.1, SATV-weight $\lambda$
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0-Iteration of SATV
    171.693912127
    51.2
     Early stopping.
    1-Iteration of SATV
    62.1600287167
    51.2
     Early stopping.
    2-Iteration of SATV
    56.6030122109
    51.2
     Early stopping.




...


.. code-block:: default

    lam = 0.3
    satv_obj = SATV(domain_shape=image.shape,
                    reg_mode='tgv',
                    lam=lam,
                    plot_iteration=False,
                    tau='auto',
                    alpha=(0.3, 0.6),
                    noise_sigma=noise_sigma,
                    assessment=noise_sigma*np.sqrt(np.prod(image.shape)))
    satv_solution = satv_obj.solve(noisy_image, max_iter=5000, tol=1e-4)

    f = plt.figure(figsize=(9, 3))
    f.add_subplot(1, 3, 1)
    plt.gray()
    plt.axis('off')
    plt.imshow(noisy_image, vmin=0, vmax=np.max(image))
    plt.title("Noisy - PSNR: "+str(psnr(image, noisy_image)))
    f.add_subplot(1, 3, 2)
    plt.gray()
    plt.imshow(satv_solution, vmin=0, vmax=np.max(image))
    plt.title("SATGV - PSNR: "+str(psnr(image, satv_solution)))
    plt.axis('off')
    f.add_subplot(1, 3, 3)
    plt.gray()
    plt.imshow(np.reshape(satv_obj.lam, image.shape))
    plt.title("SATGV-weight $\lambda$")
    plt.axis('off')
    plt.show()



.. image:: /tutorials/images/sphx_glr_spatially_adapted_total_variation_003.png
    :alt: Noisy - PSNR: 13.99, SATGV - PSNR: 22.66, SATGV-weight $\lambda$
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0-Iteration of SATV
    171.693912127
    51.2
    0.0181177667598
    0.00681895822641
    0.00358816114008
    0.0021949561867
    0.00151712062041
    0.00109739900666
    0.000864223846837
    0.000774919550898
    0.000664707182746
    0.000666144848679
    0.000565197866876
    0.000465394394756
    0.000435155460224
    0.000376188852254
    0.000336749790499
    0.000305606333372
    1-Iteration of SATV
    63.9680952684
    51.2
    0.00719013184408
    0.00284897706062
    0.00167790049279
    0.00115529455053
    0.000886480296806
    0.00069953633358
    0.000599418077365
    0.000491691705493
    0.000418458246237
    0.000376070917757
    0.000326865536687
    0.000284421164466
    0.000253487670888
    0.000244422386638
    0.000230147732331
    0.000224991415727
    2-Iteration of SATV
    57.6239833032
    51.2
    0.00655531943155
    0.00288742757745
    0.00165898304734
    0.00112047379036
    0.000825891946648
    0.000673266727242
    0.000513183236464
    0.000434341984155
    0.000356690642513
    0.000321330349693
    0.000274482097361
    0.000228557650628
    0.000200565405497
    0.000179265276754
    0.000169702465718
    0.000152352754659
    3-Iteration of SATV
    51.4735271373
    51.2
    0.00350993140609
    0.00121272011399
    0.000636771532763
    0.000396918051784
    0.000275737584399
    0.000203552060157
    0.000156807263247
    0.000120455297774
    9.76794112007e-05
    8.15937649295e-05
    7.08915674187e-05
    6.01392863394e-05
    5.19300757252e-05
    4.63371028841e-05
    4.26855879841e-05
    4.03706873443e-05





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 8 minutes  30.121 seconds)


.. _sphx_glr_download_tutorials_spatially_adapted_total_variation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: spatially_adapted_total_variation.py <spatially_adapted_total_variation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: spatially_adapted_total_variation.ipynb <spatially_adapted_total_variation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
