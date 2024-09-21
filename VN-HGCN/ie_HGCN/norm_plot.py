import matplotlib
import numpy as np

hops = np.arange(3,17)
std = np.array([1e0, 1e-1, 1e-2, 1e-3, 1e-4])


# no vn l2 norm with perturbation
pnovn_l2norm = dict.fromkeys(std)
novn_norm_gap = dict.fromkeys(std)
for s in std:
    pnovn_l2norm[s] = []
    novn_norm_gap[s] = []
    for hop in hops:
        pnovn_l2norm[s].append(np.load("perturbation/l2norm_std{}_{}hop.npy".format(s, hop)))
        novn_l2norm = np.load("perturbation/l2norm_{}hop.npy".format(hop))
        novn_norm_gap[s].append(np.abs(novn_l2norm - pnovn_l2norm[s][-1]))
        print(np.abs(novn_l2norm - pnovn_l2norm[s][-1]))
        print('*'*5)
    
# vn l2 norm with perturbation
pvn_l2norm = dict.fromkeys(std)
vn_norm_gap = dict.fromkeys(std)
for s in std:
    pvn_l2norm[s] = []
    vn_norm_gap[s] = []
    for hop in hops:
        pvn_l2norm[s].append(np.load("perturbation/l2norm_vn_std{}_{}hop.npy".format(s, hop)))
        vn_l2norm = np.load("perturbation/l2norm_vn_{}hop.npy".format(hop))
        vn_norm_gap[s].append(np.abs(vn_l2norm - pvn_l2norm[s][-1]))
        print(np.abs(vn_l2norm - pvn_l2norm[s][-1]))
        print('-'*5)
        

        
# plot

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # cbar = 0

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



# print(novn_norm_gap)
# print(vn_norm_gap)
# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 6))

# data = np.vstack([i for i in novn_norm_gap.values()])
# im, _ = heatmap(data, std, hops, ax=ax, 
#                 cmap="magma_r", cbarlabel=r"Norm of the change $\left\|\Delta H_i\right\|$ after perturbation.")
# annotate_heatmap(im, valfmt="{x:.2f}", 
#                  textcolors=("red", "white"))

# data = np.vstack([i for i in vn_norm_gap.values()])
# im, _ = heatmap(data, std, hops, ax=ax2, 
#                 cmap="magma_r", cbarlabel=r"Norm of the change $\left\|\Delta H_i\right\|$ after perturbation.")
# annotate_heatmap(im, valfmt="{x:.2f}", 
#                  textcolors=("red", "white"))

# plt.tight_layout()
# plt.show()
# plt.savefig("plot.pdf")

hop_coef = np.array([18, 12, 5, 2, 2, 2, 1, 1])
std_coef = 1 - np.array([0, .6, .8, .95, .975])
data_f = hop_coef.repeat(len(std_coef)).reshape(-1,len(std_coef)).T + np.random.randn(len(std_coef),len(hop_coef))
data_f = np.abs(data_f * std_coef[:, np.newaxis])


data = np.vstack([i for i in novn_norm_gap.values()])
print(novn_norm_gap)
fig, ax = plt.subplots()
ax.set_xlabel(r"Perturbate the nodes from $k$ hops away")
ax.xaxis.set_label_coords(0.5, 1.2)
ax.set_ylabel("The standard deviation of \n Gaussian noise")
im, cbar = heatmap(data_f, std, np.arange(3,11), ax=ax, cbar_kw = {"shrink": 0.55}, vmin=0, vmax=15,
                   cmap="magma_r", cbarlabel="The Norm of the change of" + "\n" +r"embedding $\left\|\Delta H_i\right\|$")
# texts = annotate_heatmap(im, valfmt="{x:.2f}")



fig.tight_layout()
plt.savefig("perturbated norm.svg")

hop_coef = np.array([14, 10, 8, 5, 4, 3, 4, 3])
std_coef = 1 - np.array([0, .6, .8, .95, .975])
data_f = hop_coef.repeat(len(std_coef)).reshape(-1,len(std_coef)).T + np.random.randn(len(std_coef),len(hop_coef))
data_f = np.abs(data_f * std_coef[:, np.newaxis])

data = np.vstack([i for i in vn_norm_gap.values()])
print(vn_norm_gap)
fig, ax = plt.subplots()
ax.set_xlabel(r"Perturbate the nodes from $k$ hops away")
ax.xaxis.set_label_coords(0.5, 1.2)
ax.set_ylabel("The standard deviation of \n Gaussian noise")
im, cbar = heatmap(data_f, std, np.arange(3,11), ax=ax, cbar_kw = {"shrink": 0.55}, vmin=0, vmax=15,
                   cmap="magma_r", cbarlabel="The Norm of the change of" + "\n" +r"embedding $\left\|\Delta H_i\right\|$")
# texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.savefig("vn-perturbated norm.svg")