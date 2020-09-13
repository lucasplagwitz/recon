.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_2d_image_smoothing.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorials_2d_image_smoothing.py:


01. Smoothing
================
This example ...

We import ....


.. code-block:: default


    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import misc

    from recon.interfaces import Smoothing, SmoothBregman

    img = misc.ascent()
    img = img/np.max(img)
    gt = img

    vmin, vmax = 0, 1

    # create noisy image
    sigma = 0.2
    n = np.random.normal(0, sigma, gt.shape)
    noise_img = gt + n

    f = plt.figure(figsize=(6, 3))
    plt.gray()
    f.add_subplot(1,2, 1)
    plt.title("GT")
    plt.axis('off')
    plt.imshow(gt, vmin=vmin, vmax=vmax)
    f.add_subplot(1, 2, 2)
    plt.gray()
    plt.title("Noisy")
    plt.imshow(noise_img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.show(block=False)




.. image:: /tutorials/images/sphx_glr_2d_image_smoothing_001.png
    :alt: GT, Noisy
    :class: sphx-glr-single-img





TV-Regularization and Tikhonov





.. code-block:: default


    # TV smoothing small alpha
    tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', alpha=0.3)
    u_tv = tv_smoothing.solve(data=noise_img, max_iter=450, tol=10**(-5))

    # Tikhonov smoothing -> with lam = 1 => alpha > 1 we decrease lam instead.
    tikh_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tikhonov', lam=0.1, alpha=1, tau=0.1)
    u_tik = tikh_smoothing.solve(data=noise_img, max_iter=450, tol=10**(-5))

    f = plt.figure(figsize=(6, 3))
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.gray()
    plt.imshow(u_tik, vmin=vmin, vmax=vmax)
    plt.title("Tikhonov")
    f.add_subplot(1, 2, 2)
    plt.imshow(u_tv, vmin=vmin, vmax=vmax)
    plt.title("TV")
    plt.axis('off')
    plt.gray()
    plt.show(block=False)




.. image:: /tutorials/images/sphx_glr_2d_image_smoothing_002.png
    :alt: Tikhonov, TV
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Primal-Dual Algorithm: [                                        ]--------------------]
    Primal-Dual Algorithm: [                                        ]-----]
    early stopping!



1D compare with [gt, noise, tikhonov, tv]


.. code-block:: default


    x_min = 84
    x_max = 155
    y = 20
    plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
    plt.plot(range(x_min, x_max), u_tik[x_min:x_max,y], color="blue", label="Tikhonov")
    plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
    plt.plot(range(x_min, x_max), u_tv[x_min:x_max,y], color="green", label="TV")
    plt.legend(loc="lower left")
    plt.plot(bbox_inches='tight', pad_inches=0)
    plt.show()




.. image:: /tutorials/images/sphx_glr_2d_image_smoothing_003.png
    :alt: 2d image smoothing
    :class: sphx-glr-single-img





Bregman ... iteration


.. code-block:: default


    breg_smoothing = SmoothBregman(domain_shape=gt.shape,
                                   reg_mode='tv',
                                   alpha=1.1,
                                   lam=1,
                                   tau=0.3,
                                   plot_iteration=False,
                                   assessment=sigma * np.sqrt(np.prod(gt.shape)))

    u_breg = breg_smoothing.solve(data=noise_img, max_iter=350, tol=5*10**(-6))

    f = plt.figure(figsize=(6, 3))
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.gray()
    plt.imshow(u_tv, vmin=vmin, vmax=vmax)
    plt.title("TV")
    f.add_subplot(1, 2, 2)
    plt.imshow(u_breg, vmin=vmin, vmax=vmax)
    plt.title("TV-Bregman")
    plt.axis('off')
    plt.gray()
    plt.show(block=False)




.. image:: /tutorials/images/sphx_glr_2d_image_smoothing_004.png
    :alt: TV, TV-Bregman
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    current norm error: 225.79566982354456
    runs till norm <: 102.4
    Primal-Dual Algorithm: [                                        ]--------------------]
    current norm error: 117.69316038636049
    runs till norm <: 102.4
    Primal-Dual Algorithm: [                                        ]--------------------]
    current norm error: 108.82520706984445
    runs till norm <: 102.4
    Primal-Dual Algorithm: [                                        ]--------------------]
    current norm error: 104.63215176543137
    runs till norm <: 102.4
    Primal-Dual Algorithm: [                                        ]--------------------]




1d comparisson with [gt, noise, bregman_tv, tv, tikhonov]


.. code-block:: default

    x_min = 84
    x_max = 155
    y = 20
    plt.plot(range(x_min, x_max), u_tik[x_min:x_max,y], color="darkcyan", label="Tikhonov")
    plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
    plt.plot(range(x_min, x_max), u_tv[x_min:x_max,y], color="green", label="TV")
    plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
    plt.plot(range(x_min, x_max), u_breg[x_min:x_max,y], color="blue", label="BregTV")
    plt.legend(loc="lower left")
    plt.show()
    plt.close()







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  28.955 seconds)


.. _sphx_glr_download_tutorials_2d_image_smoothing.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: 2d_image_smoothing.py <2d_image_smoothing.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: 2d_image_smoothing.ipynb <2d_image_smoothing.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
