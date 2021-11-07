import ParticleClass as pc
import numpy as np

import os
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(999)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir):
    os.makedirs(Snapshot_output_dir)

Conf_output_dir = './ConfsMonomers'
Path_ToConfiguration = Conf_output_dir+'/FinalMonomerConf.p'
if os.path.isfile(Path_ToConfiguration):
    '''Initialize system from existing file'''
    mols = pc.Monomers(FilePath=Path_ToConfiguration)
else:
    '''Initialize system with following parameters'''
    # create directory if it does not exist
    if not os.path.exists(Conf_output_dir):
        os.makedirs(Conf_output_dir)
    # define parameters
    NumberOfMonomers = 200
    L_xMin, L_xMax = 0, 10
    L_yMin, L_yMax = 0, 5
    lockdown_position = 3
    lockdown_opening = 1
    wall_thickness = 0.5
    NumberMono_per_kind = np.array([NumberOfMonomers])
    Radiai_per_kind = np.array([0.05])
    Densities_per_kind = np.ones(len(NumberMono_per_kind))
    k_BT = 0.005
    # call constructor, which should initialize the configuration
    mols = pc.Monomers(NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax,
                       NumberMono_per_kind, Radiai_per_kind,
                       Densities_per_kind, k_BT, lockdown_position,
                       wall_thickness)

mols.snapshot(FileName=Snapshot_output_dir + '/InitialConf.png',
              Title='$t = 0$')
# we could initialize next_event, but it's not necessary
# next_event = pc.CollisionEvent( Type = 'wall or other, to be determined',
# dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
t = 0.0
dt = 0.02
NumberOfFrames = 1000
next_event = mols.compute_next_event(t)

health_plot_t = [0]
health_plot_sick = [5]
health_plot_healthy = [200]
health_plot_healed = [200]


def MolecularDynamicsLoop(frame):
    '''
    The MD loop including update of frame for animation.
    '''
    global t, mols, next_event

    next_frame_t = t + dt

    while t + next_event.dt < next_frame_t:
        mols.pos += mols.vel * next_event.dt
        t += next_event.dt
        mols.compute_new_velocities(next_event)
        mols.compute_new_health_state(next_event, t)
        next_event = mols.compute_next_event(t)

    dt_remaining = next_frame_t - t
    mols.pos += mols.vel * dt_remaining
    t += dt_remaining
    next_event = mols.compute_next_event(t)
    health_plot_t.append(t)
    health_plot_healthy.append(200)
    line_healthy.set_data(health_plot_t, health_plot_healthy)
    f_healthy = ax[0].fill_between(health_plot_t, health_plot_healthy,
                                   color="#AAC6CA")
    health_plot_sick.append(np.count_nonzero(mols.health_state == 1))
    line_sick.set_data(health_plot_t, health_plot_sick)
    f_sick = ax[0].fill_between(health_plot_t, health_plot_sick,
                                color="#CC5756")
    health_plot_healed.append(200 - np.count_nonzero(mols.health_state == 2))
    line_healed.set_data(health_plot_t, health_plot_healed)
    f_healed = ax[0].fill_between(health_plot_t, health_plot_healed,
                                  health_plot_healthy, color="#CCC18D")

    # --> write your event-driven loop
    # --> we want a simulation in constant time steps dt

    # we can save additional snapshots for debugging -> slows down real-time
    # animation
    # mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_0.png' % t,
    # Title = '$t = %.8f$' % t)
    plt.title(f"$t = {t}$, remaining frames = {NumberOfFrames-(frame+1)}")
    colors = list(map(lambda x: "#AAC6CA"
                      if x == 0 else ("#CC5756"
                                      if x == 1 else "#CCC18D"),
                      mols.health_state))
    collection.set_color(colors)
    collection.set_offsets(mols.pos)
    return (collection, line_healthy, line_sick, line_healed,
            f_healthy, f_sick, f_healed)


'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [1, 4]})
L_xMin, L_yMin = mols.BoxLimMin  # not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax  # not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax[1].set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax[1].set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax[1].set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin, L_yMin), L_xMax-L_xMin, L_yMax-L_yMin,
                          linestyle='dashed', ec='gray', fc='None')
wall_down = mpatches.Rectangle((lockdown_position - wall_thickness/2, L_yMin),
                               wall_thickness,
                               (L_yMax - L_yMin - lockdown_opening)/2,
                               linestyle="dashed", ec="gray", fc="none")
wall_up = mpatches.Rectangle((lockdown_position - wall_thickness/2, L_yMax),
                             wall_thickness,
                             -(L_yMax - L_yMin - lockdown_opening)/2,
                             linestyle="dashed", ec="gray", fc="none")
ax[1].add_patch(rect)
ax[1].add_patch(wall_down)
ax[1].add_patch(wall_up)


# plotting all monomers as solid circles of individual color
MonomerColors = np.array(["#000000"]*mols.NM)
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x',
                               offsets=mols.pos, transOffset=ax[1].transData,
                               cmap='nipy_spectral', edgecolor='k')
collection.set_color(MonomerColors)
collection.set_clim(0, 1)  # <--- we set the limit for the color code
ax[1].add_collection(collection)

line_healthy, = ax[0].plot(health_plot_t, health_plot_healthy, color="#AAC6CA")
line_sick, = ax[0].plot(health_plot_t, health_plot_sick, color="#CC5756")
line_healed, = ax[0].plot(health_plot_t, health_plot_healed, color="#CCC18D")
ax[0].set_xlim(0, NumberOfFrames*dt)
ax[0].set_ylim(0, NumberOfMonomers)

'''
Create the animation, i.e. looping NumberOfFrames over the update function
'''
Delay_in_ms = 33.3  # dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames,
                    interval=Delay_in_ms, blit=False, repeat=False)
plt.show()

'''Save the final configuration and make a snapshot.'''
# write the function to save the final configuration
mols.save_configuration(Path_ToConfiguration)
mols.snapshot(FileName=f"{Snapshot_output_dir}/FinalConf.png",
              Title=f"$t = {t:.4f}$")
