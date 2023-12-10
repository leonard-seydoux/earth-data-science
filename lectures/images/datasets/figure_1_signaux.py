#!/Users/seydoux/anaconda3/envs/covseisnet/bin/python
# coding: utf-8
"""Figure 1: diversité de signaux volcaniques.

Figures extraites de Moran et al. (2008) et Zobin (2014).
"""

import obspy

from cartopy import crs, feature
from matplotlib import pyplot as plt
from os import path

ROOT_PATH = (
    "/Users/seydoux/Dropbox/Applications/Overleaf/Dossier CNRS 2022/Rapport/"
)
FIGURE_SIZE = 5.91, 3.2
FIGURE_PATH = "./"

STREAM_PATH = "/Users/seydoux/Data/JUN.HHZ/con*/processed/*EHZ"
STREAM_SEGMENT = 7200

IMAGE_PATH = ROOT_PATH + "/sources/zobin"


def main():

    # FIGURE
    # ------

    # settings
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["ytick.left"] = False

    # generate
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(6, 2, hspace=2)
    grid = grid[:, 0], *[grid[i, 1] for i in range(6)]
    ax = [fig.add_subplot(g) for g in grid]

    # labels
    ax[0].set_title("a", loc="left", pad=15)
    ax[1].set_title("b", loc="left", pad=15)

    # SEISMOGRAMS
    # -----------

    # read
    stream = obspy.read(STREAM_PATH)

    # process
    trace = stream.merge(1)[0]
    trace = trace.decimate(2)
    trace_slide = trace.slide(STREAM_SEGMENT, STREAM_SEGMENT)

    # show
    for index, segment in enumerate(trace_slide):
        t = segment.times() / 60
        x = segment.data / 0.5e5
        color = "#3240a8" if (index % 2 == 0) else "#3285a8"
        day = STREAM_SEGMENT * index / 24 / 3600
        ax[0].plot(t, day + x, rasterized=True, lw=0.1, c=color)

    # labelize
    ax[0].tick_params(axis="y", length=0)
    ax[0].set_ylim(-0.0, day + 0.2)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 120)
    ticks = range(0, 121, 40)
    labels = [f"{t}" for t in ticks]
    labels[-1] += " mn"
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(labels)
    # ax[0].set_xlabel("Time (minutes)")
    ax[0].set_ylabel("Vertical ground speed (days since 2004/09/27)")

    # annotate
    ax[0].axvline(x=122, ymin=0.45, ymax=0.88, c="C1", clip_on=False)
    ann = "Vocano-\ntectonic\nand hybrid\nearthquakes"
    ax[0].text(124, 1, ann, c="C1", size="small", va="top")

    # wind
    ax[0].fill_between([33, 80], 0.55, 0.75, fc="none", ec="C0", zorder=10)
    ann = "Wind"
    ax[0].text(122, 0.68, ann, c="C0", size="small", va="center")

    # wind
    ax[0].fill_between([30, 45], 7.6, 7.7, fc="w", ec="0.6", zorder=10)
    ann = "Gap"
    ax[0].text(122, 7.65, ann, c="0.6", size="small", va="center")

    # exp
    ax[0].fill_between([72, 88], 4.65, 4.85, fc="none", ec="C1", zorder=10)
    ann = "Explosion"
    ax[0].text(122, 4.75, ann, c="C1", size="small", va="center")

    # tremor
    ax[0].fill_between(
        [75, 120], 5.6, 5.9, fc="none", ec="C1", zorder=10, clip_on=False
    )
    ann = "Tremor"
    ax[0].text(122, 5.75, ann, c="C1", size="small", va="center")

    # globe
    pos = -122, 46
    projection = crs.Mollweide(central_longitude=pos[0])
    p = ax[0].get_position()
    position = [p.x1 - 0.15, p.y1 + 0.01, 0.15, 0.15]
    ax_map = fig.add_axes(position, projection=projection)
    ax_map.set_global()
    ax_map.add_feature(feature.LAND, fc="0.85", lw=0)
    ax_map.plot(*pos, "C1s", transform=crs.PlateCarree())
    label = "Mt. Saint Helens"
    ax_map.text(0, 0, label, ha="center", size="small", va="center")

    # ZOOMS
    # -----

    t_short = 50

    # volcanotectonic
    short = "vt.png", "exp.png", "lp.png"
    name = "Volcano-tectonic", "Explosion", "Long-period"
    for index, (event, name) in enumerate(zip(short, name)):
        img = plt.imread(path.join(IMAGE_PATH, event))
        extent = [0, t_short, 0, 1]
        ax[index + 1].imshow(img, extent=extent)
        ax[index + 1].set_xlim([0, t_short])
        ax[index + 1].set_ylim(bottom=-0.2)
        ax[index + 1].text(0, 1, name, va="bottom", size="small", color="0.2")
        if index < 2:
            ax[index + 1].set_axis_off()
    ax[3].set_ylim(bottom=-0.4)
    ticks = range(0, t_short + 1, 10)
    labels = [f"{t}" for t in ticks]
    labels[-1] += " s"
    ax[3].set_xticks(ticks)
    ax[3].set_xticklabels(labels)
    ax[3].set_yticks([])

    # tremor
    t_trem = 120
    img = plt.imread(path.join(IMAGE_PATH, "tr.png"))
    extent = [0, t_trem, 0, 1]
    ax[4].imshow(img, extent=extent)
    ax[4].set_xlim([0, t_trem])
    ax[4].set_ylim(bottom=-0.2)
    ticks = range(0, t_trem + 1, 40)
    labels = [f"{t}" for t in ticks]
    labels[-1] += " s"
    ax[4].set_xticks(ticks)
    ax[4].set_xticklabels(labels)
    ax[4].set_yticks([])
    ax[4].text(0, 1, "Tremor", va="bottom", size="small", color="0.2")

    # pyroclastic flow
    t_pf = 4
    img = plt.imread(path.join(IMAGE_PATH, "pf.png"))
    extent = [0, t_pf, 0, 1]
    ax[5].imshow(img, extent=extent)
    ax[5].set_xlim([0, t_pf])
    ax[5].set_ylim(bottom=-0.2, top=0.9)
    ticks = range(0, t_pf + 1, 1)
    labels = [f"{t}" for t in ticks]
    labels[-1] += " mn"
    ax[5].set_xticks(ticks)
    ax[5].set_xticklabels(labels)
    ax[5].set_yticks([])
    ax[5].text(0, 1, "Pyroclastic flow", va="bottom", size="small", color="0.2")

    # lahars
    t_lah = 80
    img = plt.imread(path.join(IMAGE_PATH, "lah.png"))
    extent = [0, t_lah, 0, 1]
    ax[6].imshow(img, extent=extent)
    ax[6].set_xlim([0, t_lah])
    ax[6].set_ylim(bottom=-0.2)
    ticks = range(0, t_lah + 1, 20)
    labels = [f"{t}" for t in ticks]
    labels[-1] += " mn"
    ax[6].set_xticks(ticks)
    ax[6].set_xticklabels(labels)
    ax[6].set_yticks([])
    ax[6].text(0, 1, "Lahars", va="bottom", size="small", color="0.2")

    # globe
    pos = -103, 19
    projection = crs.Mollweide(central_longitude=pos[0])
    p = ax[1].get_position()
    position = [p.x1 - 0.07, p.y1 + 0.01, 0.15, 0.15]
    ax_map = fig.add_axes(position, projection=projection)
    ax_map.set_global()
    ax_map.add_feature(feature.LAND, fc="0.85", lw=0)
    ax_map.plot(*pos, "C1s", transform=crs.PlateCarree())
    label = "Volcan de Colima"
    ax_map.text(0, 0, label, ha="center", size="small", va="top")

    # SAVE
    # ----

    name, _ = path.splitext(path.basename(__file__))
    fig.savefig(path.join(FIGURE_PATH, name + ".png"), dpi=300)


if __name__ == "__main__":
    main()
