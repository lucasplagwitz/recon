.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_white_noise_between_spaces.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorials_white_noise_between_spaces.py:


07. White Noise Between Spaces
================

This Notebook examines the behavior of noise under linear operators numerically. Lets say for some $u,w \in H$, $K \in L(u,w)$ and $\eta$ some distributed noise:
egin{equation}
    w = Ku \quad w_\eta = Ku + \eta
\end{equation}
Is the bias $\eta_u = (K^{-1}w- K^{-1}w_\eta)$ distributed?

We import ....


.. code-block:: default


    import numpy as np
    from scipy import misc
    import matplotlib.pyplot as plt

    from recon.operator.ct_radon import CtRt

    u = misc.face(gray=True)[256:256*3, 256:256*3]
    u = u/np.max(u)
    plt.imshow(u, cmap=plt.cm.gray)
    plt.show()





.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_001.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img





2


.. code-block:: default


    mu, sigma = 0, 0.1
    eta_image = np.random.normal(mu, sigma, u.shape)
    plt.imshow(u+eta_image, cmap=plt.cm.gray)
    plt.show()



.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_002.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img






.. code-block:: default

    count, bins, ignored = plt.hist(eta_image.ravel(), 50, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.show()




.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_003.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img





3


.. code-block:: default

    theta = list(np.linspace(0., 180., 180, endpoint=False))

    R = CtRt(np.shape(u),
             np.array([(np.shape(u)[0]/2)+1, (np.shape(u)[0]/2)+1]),
             theta=theta)
    w = R*u.ravel()
    w0 = R*(u+eta_image).ravel()
    eta_w = w-w0

    count, bins, ignored = plt.hist(eta_w.ravel(), 50, density=True)
    plt.show()




.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_004.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img





# Backward


.. code-block:: default


    mu, sigma = 0, 0.01*np.max(w)
    eta = np.random.normal(mu, sigma, w.shape)
    w_eta = w + eta
    plt.imshow(np.reshape(R.inv*w_eta.ravel(), u.shape), cmap=plt.cm.gray)
    plt.show()



.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_005.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img






.. code-block:: default

    recon_eta_est = (np.reshape(R.inv*w_eta.ravel(), u.shape)-u).ravel()
    count, bins, ignored = plt.hist(recon_eta_est, 50, density=True)
    plt.show()



.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_006.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img






.. code-block:: default

    print("Backwards-Image-Mean: "+str(round(np.mean(recon_eta_est), 4)))
    print("Backwards-Image-Sigma: "+str(round(np.std(recon_eta_est, ddof=1), 4)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Backwards-Image-Mean: 0.0018
    Backwards-Image-Sigma: 0.1938




# Only Noise


.. code-block:: default


    mu, sigma = 0, 0.01*np.max(w)
    eta = np.random.normal(mu, sigma, w.shape)
    eta_est = R.inv*eta.ravel()
    sigma_est = np.std(eta_est, ddof=1)
    mu_est = np.mean(eta_est)
    count, bins, ignored = plt.hist(eta_est, 50, density=True)
    plt.plot(bins, 1/(sigma_est * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu_est)**2 / (2 * sigma_est**2)),
             linewidth=2, color='r')
    plt.show()
    print("Only-Noise-Recon:Mean: "+str(round(np.mean(eta_est), 4)))
    print("Only-Noise-Recon-Sigma: "+str(round(np.std(eta_est, ddof=1), 4)))




.. image:: /tutorials/images/sphx_glr_white_noise_between_spaces_007.png
    :alt: white noise between spaces
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Only-Noise-Recon:Mean: -0.0
    Only-Noise-Recon-Sigma: 0.1876




# Result


.. code-block:: default

    print("Mean Difference: "+str(np.abs(np.mean(eta_est) - np.mean(recon_eta_est))))
    print("Sigma Difference: "+str(np.abs(np.std(eta_est, ddof=1) - np.std(recon_eta_est, ddof=1))))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Mean Difference: 0.00183966989429
    Sigma Difference: 0.00621486247736




Therefore for a normal distributed $\eta \sim \mathcal N(0, \sigma_0)$:
\begin{equation}
    K^{-1}(w+\eta)-K^{-1}(w) \sim K^{-1}\eta \sim \mathcal N(0, \sigma_1)
\end{equation}

\sigma_1 has to be calculated.

# Shape independent


.. code-block:: default

    mu, sigma = 0, 0.01*np.max(w)
    for shape in [(128, 128), (256, 256), (512, 512)]:
        R = CtRt(shape,
                 np.array([(shape[0] // 2), (shape[0] // 2)]),
                 theta=theta)
        eta = np.random.normal(mu, sigma, R.image_dim)
        eta_est = R.inv*eta.ravel()
        print(str(shape) + "-Mean: "+str(round(np.mean(eta_est), 4)))
        print(str(shape) +"-Sigma: "+str(round(np.std(eta_est, ddof=1),4)))



.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (128, 128)-Mean: -0.0002
    (128, 128)-Sigma: 0.187
    (256, 256)-Mean: -0.0
    (256, 256)-Sigma: 0.1871
    (512, 512)-Mean: 0.0001
    (512, 512)-Sigma: 0.1879





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  12.607 seconds)


.. _sphx_glr_download_tutorials_white_noise_between_spaces.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: white_noise_between_spaces.py <white_noise_between_spaces.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: white_noise_between_spaces.ipynb <white_noise_between_spaces.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
