# description
# description
# description

import numpy as np
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt


from recon.math.operator.mri_dft import MriDft
from recon.math.terms import Dataterm, Projection, DatatermRecBregman, DatatermBregman
from recon.math.operator.first_derivative import FirstDerivative
from recon.math.pd_hgm import PdHgm
from recon.segmentation.tv_pdghm import multi_class_segmentation
from recon.segmentation.tv_bregman_pdghm import multi_class_segmentation_bregman
from recon.helpers.functions import normest



# - Start Setting Section -

data_import_path = "./../data/joint_recon_vcorona/"
data_output_path = data_import_path+"output/"

# load content
content = loadmat(data_import_path+'brainphantom.mat')
gt = content["gt"]
gt_seg = content["gt_seg"] - 1
content = loadmat(data_import_path+'spiralsampling.mat')
samp = content["samp"]

fig = plt.figure()
plt.Figure()
plt.imshow(gt, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.title('Groundtruth')
plt.savefig(data_output_path+'gt.png')
plt.close()

fig = plt.figure()
plt.Figure()
plt.imshow(gt_seg)
plt.axis('off')
plt.title('Groundtruth Segmentation')
plt.savefig(data_output_path+'gt_seg.png')
plt.close()

# - End Setting Section -



F = MriDft(np.shape(gt), np.array([(np.shape(gt)[0]/2)+1, (np.shape(gt)[0]/2)+1]))

samp_values = np.prod(np.shape(samp))

S = sparse.eye(samp_values)
a = samp.ravel()
a = np.array(np.where(a==1)[0])
S = S.tocsr()[a,:] # ?
nz = np.count_nonzero(samp)/samp_values  # undersampling rate
SF = S*F
g =SF*gt.ravel()

# Gaussian noise
sigma = 0.005
n = sigma*np.max(np.abs(g))*np.random.normal(size=g.shape[0])
f = g + n


# Gradient  operator
ex = np.ones((gt.shape[1],1))
ey = np.ones((1, gt.shape[0]))
dx = sparse.diags([1, -1], [0, 1], shape=(gt.shape[1], gt.shape[1])).tocsr()
dx[gt.shape[1]-1, :] = 0
dy = sparse.diags([-1, 1], [0, 1], shape=(gt.shape[0], gt.shape[0])).tocsr()
dy[gt.shape[0]-1, :] = 0

grad = sparse.vstack((sparse.kron(dx, sparse.eye(gt.shape[0]).tocsr()),
                      sparse.kron(sparse.eye(gt.shape[1]).tocsr(), dy)))


# Plot Zero-filling reconstructions
fig, axs = plt.subplots(2, 2)

#set(figure,'defaulttextinterpreter','latex');
fontdict = {'fontsize': 8}
axs[0, 0].imshow(gt)
axs[0, 0].set_title('Groundtruth', fontdict)
axs[0, 0].axis('off')

axs[0, 1].imshow(samp/np.max(samp))
axs[0, 1].set_title('Undersampling matrix', fontdict)
axs[0, 1].axis('off')

axs[1, 0].imshow(np.reshape(abs(SF.T*g), gt.shape), vmin=0, vmax=np.max(gt))
axs[1, 0].set_title('Zero-filled recon of noise-free data', fontdict)
axs[1, 0].axis('off')

axs[1, 1].imshow(np.reshape(abs(SF.T*f), gt.shape), vmin=0, vmax=np.max(gt))
axs[1, 1].set_title('Zero-filled recon of noisy data', fontdict)
axs[1, 1].axis('off')

plt.savefig(data_output_path+'beginning.png')
plt.close(fig)


# segmenatation parameter
beta0 = 0.0001  #regulatisation parameter for segmentation
segmentation_tau = 3500
c1 = 0.01
c2 = 0.3
c3 = 0.7
c4 = 0.85
classes = [c1, c2, c3, c4]

# modes
tv_recon = True
tv_bregman = True
joint_recon = True

# saved normest result for given parameters to improve run_time
recalc_normest = False

# TV regularised Reconstruction
if tv_recon:
    alpha0=0.2
    K = alpha0*grad

    if recalc_normest:
        norm = normest(K)
        sigma0=0.99/norm
        print(sigma0)
    else:
        sigma0= 1.7501
    tau0 = sigma0

    G = Dataterm(S, F)
    F_star = Projection(gt.shape)
    solver = PdHgm(K, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 300
    solver.tol = 5*10**(-4)

    G.set_proxdata(f)
    solver.solve()

    u0 = np.reshape(np.real(solver.var['x']), gt.shape)
    rel_tvrec = np.linalg.norm(gt - u0, 2) #/np.linalg.norm(gt)

    fig = plt.figure()
    plt.Figure()
    plt.imshow(u0, vmin=0, vmax=np.max(gt))
    plt.axis('off')
    plt.title('TV Reconstruction, alpha=' + str(alpha0) +',RRE ='+ str(rel_tvrec), y=-0.1)
    plt.savefig(data_output_path+'tv_reconstruction.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    # Segmentation on TV regularised Reconstruction
    # currently only one levelset supported

    tv_seg, vd1 = multi_class_segmentation(u0, classes=classes, beta=beta0, tau=segmentation_tau)


    fig = plt.figure()
    plt.Figure()
    plt.imshow(tv_seg)
    PRE = len(np.where( (gt_seg != tv_seg)) [0]) / (np.prod(gt_seg.shape))
    plt.axis('off')
    plt.title('Segmentation on Breg reconstruction, beta='+ str(beta0) +', PRE ='+ str(PRE), y=-0.1)
    plt.savefig(data_output_path+"tv_segmentation.png")
    plt.close(fig)


# Bregman TV Reconstruction
if tv_bregman:
    print("Bregman reconstruction")
    alpha01=1.1  # regularisation parameter
    K01 = alpha01*grad

    if recalc_normest:
        norm = normest(K01)
        sigma0 = 0.99 / norm
        print(sigma0)
    else:
        sigma0 = 0.3182
    tau0 = sigma0
    pk = np.zeros(gt.shape)
    pk = pk.T.ravel()

    plt.Figure()
    ulast = np.zeros(gt.shape)
    u01=ulast
    i=0
    plot_pre_iter = False

    while (np.linalg.norm(SF * u01.ravel() - f.T.ravel(), 2) > 0.005 * np.max(np.abs(g)) * np.sqrt(np.prod(f.shape))):
        ulast = u01

        G = DatatermRecBregman(S, F)
        F_star = Projection(gt.shape)

        solver = PdHgm(K01, F_star, G)
        G.set_proxparam(tau0)
        F_star.set_proxparam(sigma0)
        solver.maxiter = 150
        solver.tol = 5 * 10**(-4)

        G.set_proxdata(f)
        G.setP(pk)
        solver.solve()
        u01 = np.reshape(np.real(solver.var['x']), gt.shape)
        pklast = pk
        pk = pk - (1/alpha01) * (np.real(F.T*S.T*( SF * u01.ravel() -f)))
        i=i+1

        RRE_breg = np.linalg.norm(gt - u01, 2)
        if plot_pre_iter:
            plt.imshow(u01, vmin=0, vmax=np.max(gt))
            plt.axis('off')
            plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)

            plt.savefig(data_output_path+'Bregman_reconstruction_iter'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    fig = plt.figure()
    u0B=u01
    RRE_breg=np.linalg.norm(gt - u0B, 2) #/np.linalg.norm(gt)
    plt.imshow(u0B, vmin=0, vmax=np.max(gt) )
    plt.axis('off')
    plt.title('Bregman Reconstruction, alpha='+ str(alpha01) +', RRE =' + str(RRE_breg), y=-0.1)

    plt.savefig(data_output_path+'Bregman_reconstruction.png')
    plt.close(fig)


    vb, _ = multi_class_segmentation(u0B, classes=classes, beta=beta0, tau=segmentation_tau)

    PRE_breg = len(np.where( (gt_seg != vb)) [0]) / (np.prod(gt_seg.shape))

    fig = plt.figure()
    plt.imshow(vb)
    plt.axis('off')
    plt.title('Segmentation on Breg reconstruction, beta='+ str(beta0) +', PRE ='+ str(PRE_breg), y=-0.1)
    plt.savefig(data_output_path+'bregman_segmentation.png')
    plt.close(fig)


# Joint reconstruction and segmentation
# Parameter initialization
if joint_recon:
    print("Joint reconstruction and segmentation")

    alpha = 0.8
    beta = 10e-7
    segmentation_tau_bregman = 350024
    delta = 0.001

    K1 = alpha * grad
    if recalc_normest:
        sigma = 0.99 / normest(K1)
        print(sigma)
    else:
        sigma = 0.4375
    tau = sigma

    y0 = 0.01

    v = tv_seg

    V = 0
    V1=0
    vd = vd1
    for i in range(len(classes)):
        V = V + vd[:,:,i]
        V1= V1+ vd[:,:,i] * classes[i]


    A = delta*V.ravel()
    B = delta*V1.ravel()


    N=6

    rec = np.zeros((256, 256, N))
    seg = np.zeros((256, 256, N))

    rec[:,:,0] = u0
    rec[:,:,1] = np.zeros(gt.shape)
    seg[:,:,0] = np.reshape(v, gt.shape)
    u = np.zeros(gt.shape)

    # Joint reconstruction and segmentation
    pk = np.zeros(gt.shape)
    pk = pk.ravel()
    qk = np.zeros((np.prod(gt.shape), len(classes)))

    rel_rec_error = []
    rel_seg_error = []

    vlast = 0

    # Plot Zero-filling reconstructions
    fig, axs = plt.subplots(N, 2)

    i = 0
    last = False

    while (i < N): # (np.linalg.norm(SF * u.ravel() - f.T.ravel()) > 0.005*max(abs(g)) * np.sqrt(np.prod(f.shape))):
        #print(np.linalg.norm(SF * u.ravel() - f.T.ravel()))
        #print(np.linalg.norm(n))
        #print(0.005*max(abs(g)) * np.sqrt(np.prod(f.shape)))
        #print(np.linalg.norm(vd-vlast))
        solver1 = None
        v = v.ravel()
        V = 0
        V1 = 0
        for m in range(len(classes)):
            V = V + vd[:, :, m]
            V1 = V1 + vd[:, :, m] * classes[m]

        A = delta * V.ravel()
        B = delta * V1.ravel()


        Gu = DatatermBregman(S, F, A, B)
        Fu_star = Projection(gt.shape)
        Gu.set_proxparam1(alpha)
        Gu.set_proxdata(f)
        Gu.set_proxparam(tau)

        Fu_star.set_proxparam(sigma)
        Gu.setP(pk)

        solver1 = PdHgm(K1, Fu_star, Gu)
        solver1.maxiter = 150
        solver1.tol = 5 * 10**(-4)
        solver1.sens = 0.001
        solver1.solve()

        u = np.reshape(np.real(solver1.var['x']), gt.shape)


        pk = pk - (1 / alpha) * (np.real(F.T*S.T * (SF * u.ravel() - f)) + 4/2 * A * u.ravel() - 4/2 * B )

        rec[:,:, i] = u
        rel_rec_error.append(np.linalg.norm(gt - u, 2)) # / np.linalg.norm(gt))

        vlast = vd
        delta1 = delta

        v, vd = multi_class_segmentation_bregman(u, classes, beta , delta1, qk, segmentation_tau_bregman)


        qk[:, 0] = qk[:, 0] - (1 / beta) * delta1 * ((u.ravel() - classes[0]))** 2
        qk[:, 1] = qk[:, 1] - (1 / beta) * delta1 * ((u.ravel() - classes[1]))** 2
        qk[:, 2] = qk[:, 2] - (1 / beta) * delta1 * ((u.ravel() - classes[2]))** 2
        qk[:, 3] = qk[:, 3] - (1 / beta) * delta1 * ((u.ravel() - classes[3]))** 2

        rel_seg_error.append(len(np.where( (gt_seg != v)) [0]) / (np.prod(gt_seg.shape)))
        print(rel_seg_error)
        print(rel_rec_error)
        seg[:, :, i] = v

        axs[i, 0].imshow(u)
        axs[i, 0].set_title('M= '+str(rel_rec_error[i]))
        axs[i, 0].axis('off')

        axs[i, 1].imshow(v)
        axs[i, 1].set_title('M= ' + str(rel_seg_error[i]))
        axs[i, 1].axis('off')

        i += 1

    plt.savefig(data_output_path+'joint_recon_iterations.png')
    plt.close(fig)

    fig = plt.figure()
    plt.Figure()
    plt.imshow(u, vmin=0, vmax=np.max(gt))
    plt.axis('off')
    plt.title('Joint, RRE ='+ str(rel_rec_error[-1]), y=-0.1)
    plt.savefig(data_output_path+'joint_reconstruction.png')
    plt.close()

    fig = plt.figure()
    plt.Figure()
    plt.imshow(v)
    plt.axis('off')
    plt.title('Joint seg, PRE ='+ str(rel_seg_error[-1]), y=-0.1)
    plt.savefig(data_output_path+'joint_segmentation.png')
    plt.close()



