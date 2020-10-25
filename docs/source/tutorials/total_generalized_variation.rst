.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_total_generalized_variation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorials_total_generalized_variation.py:


04. Total Generalized Variation
===============================
Now we take a step deeper into Total Variation based regularization.

We focus on concepts from different papers.
Mainly we use for numerical access:
    Knoll, Bredis, Pock: Second Order Total Generalized Variation (TGV) for MRI

The first order Total Variation got some problems with smooth edges.
See following noisy example with the TV-Regularization.


.. code-block:: default

    import numpy as np
    import matplotlib.pyplot as plt

    from recon.utils import psnr
    from recon.utils.images import two_smooth_squares
    from recon.interfaces import Smoothing, SmoothBregman

    image = two_smooth_squares(256, 128)
    noise_image = image + np.random.normal(0, 0.2*np.max(image), size=image.shape)

    tv_denoising = Smoothing(domain_shape=image.shape, reg_mode='tv', lam=0.3, alpha=0.1, tau='calc')
    tv_solution = tv_denoising.solve(noise_image, max_iter=2000, tol=1e-4)

    f = plt.figure(figsize=(6, 3))
    f.add_subplot(1, 2, 1)
    plt.gray()
    plt.axis('off')
    plt.imshow(noise_image, vmin=0, vmax=np.max(image))
    plt.title("Noisy")
    f.add_subplot(1, 2, 2)
    plt.gray()
    plt.imshow(tv_solution, vmin=0, vmax=np.max(image))
    plt.title("TV based denoising")
    plt.axis('off')
    plt.show()




.. image:: /tutorials/images/sphx_glr_total_generalized_variation_001.png
    :alt: Noisy, TV based denoising
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

     Early stopping.




To avoid strong stair-casing effects, we introduce the Total Generalized Variation.
At this point there is no interface for second order TV. We implement it direct with the
adapted Primal-Dual algorithm.


.. code-block:: default


    from recon.solver.pd_hgm_extend import PdHgmTGV

    # TGV smoothing small alpha
    alpha = (0.3, 0.6)
    solver = PdHgmTGV(alpha=alpha, lam=0.9)
    tgv_solution = np.reshape(solver.solve(noise_image), image.shape)

    f = plt.figure(figsize=(9, 3))
    f.add_subplot(1, 3, 1)
    plt.gray()
    plt.axis('off')
    plt.imshow(image, vmin=0, vmax=np.max(image))
    plt.title("Original")
    f.add_subplot(1, 3, 2)
    plt.gray()
    plt.axis('off')
    plt.imshow(tv_solution, vmin=0, vmax=np.max(image))
    plt.title("TV based denoising")
    f.add_subplot(1, 3, 3)
    plt.gray()
    plt.imshow(tgv_solution, vmin=0, vmax=np.max(image))
    plt.title("TGV based denoising")
    plt.axis('off')
    plt.show()





.. image:: /tutorials/images/sphx_glr_total_generalized_variation_002.png
    :alt: Original, TV based denoising, TGV based denoising
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.00771069298652
    0.0031111419832
    0.00180729507596
    0.00122331678004
    0.000907678083309
    0.000728211463402
    0.000608057434836
    0.000504880376087
    0.000440147529181
    0.000372074371319




Since TGV also represents a convex functional, it can also be extended by Bregman.
Maybe there will be an interface for this in the future.


.. code-block:: default


    plot_iteration = False
    lam = 0.3
    assessment = 0.2 * np.max(image) * np.sqrt(np.prod(noise_image.shape))
    pk = np.zeros(image.shape)
    pk = pk.ravel()
    i = 0

    u = np.zeros(image.shape)
    while True:
        print("current norm error: " + str(np.linalg.norm(u.ravel() - noise_image.ravel(), 2)))
        print("runs till norm <: " + str(assessment))

        solver = PdHgmTGV(alpha=alpha, lam=lam, mode='tgv', pk=pk)

        u_new = np.reshape(solver.solve(noise_image), image.shape)

        if np.linalg.norm(u_new.ravel() - noise_image.ravel(), 2) < assessment:
            break

        u = u_new
        pk = pk - lam / alpha[0] * (u.ravel() - noise_image.ravel())
        i = i + 1

        if plot_iteration:
            plt.gray()
            plt.imshow(u)
            plt.axis('off')
            plt.savefig('Bregman_TGV_iter' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    current norm error: 165.119961638
    runs till norm <: 51.2
    0.012877106605
    0.00521424351287
    0.00284649811812
    0.00176420359798
    0.0012907136069
    0.000988516676199
    0.000852329732466
    0.000816692532205
    0.000667334100981
    0.000556070797568
    current norm error: 52.9145316603
    runs till norm <: 51.2
    0.0130646688425
    0.00503696518632
    0.00302054257841
    0.00194678593056
    0.00142751246296
    0.00109094354898
    0.00089387244592
    0.000766550512147
    0.000619090179674
    0.000534423426562
    current norm error: 51.4622815594
    runs till norm <: 51.2
    0.0155674014557
    0.00591369574443
    0.00336137692153
    0.0021950467685
    0.00157342659696
    0.00118577464407
    0.000967438770227
    0.00079032204254
    0.000647887128984
    0.000564429791908
    current norm error: 51.3277683524
    runs till norm <: 51.2
    0.0182142922654
    0.00675238577248
    0.00363658070046
    0.00232872520605
    0.00162157138821
    0.00122637344611
    0.000976911132042
    0.000793198843987
    0.000664255558431
    0.00056278336667




Compare it to normal TV-Bregman


.. code-block:: default


    breg_smoothing = SmoothBregman(domain_shape=image.shape,
                                   reg_mode='tv',
                                   alpha=1,
                                   lam=0.5,
                                   tau='calc',
                                   plot_iteration=False,
                                   assessment=assessment)

    u_breg = breg_smoothing.solve(data=noise_image, max_iter=2000, tol=1e-4)


    f = plt.figure(figsize=(9, 3))
    f.add_subplot(1, 3, 1)
    plt.gray()
    plt.axis('off')
    plt.imshow(image, vmin=0, vmax=np.max(image))
    plt.title("Original")
    f.add_subplot(1, 3, 2)
    plt.gray()
    plt.axis('off')
    plt.imshow(np.reshape(u_breg, image.shape), vmin=0, vmax=np.max(image))
    plt.title("Bregman-TV ")
    f.add_subplot(1, 3, 3)
    plt.gray()
    plt.imshow(np.reshape(u, image.shape), vmin=0, vmax=np.max(image))
    plt.title("Bregman-TGV")
    plt.axis('off')
    plt.show()

    print("TV-PSNR: "+str(psnr(image, tv_solution)))
    print("TGV-PSNR: "+str(psnr(image, tgv_solution)))
    print("Bregman-TV-PSNR: "+str(psnr(image, u_breg)))
    print("Bregman-TGV-PSNR: "+str(psnr(image, u_new)))


.. image:: /tutorials/images/sphx_glr_total_generalized_variation_003.png
    :alt: Original, Bregman-TV , Bregman-TGV
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    current norm error: 165.119961638
    runs till norm <: 51.2
     Early stopping.
    0.428998886775
    current norm error: 54.3755669727
    runs till norm <: 51.2
     Early stopping.
    0.836379326847
    current norm error: 51.2498963996
    runs till norm <: 51.2
     Early stopping.
    TV-PSNR: 32.31
    TGV-PSNR: 32.85
    Bregman-TV-PSNR: 31.12
    Bregman-TGV-PSNR: 34.45





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 5 minutes  24.944 seconds)


.. _sphx_glr_download_tutorials_total_generalized_variation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: total_generalized_variation.py <total_generalized_variation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: total_generalized_variation.ipynb <total_generalized_variation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
