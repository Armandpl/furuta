# simulation params for our robot
params = dict(
    # Gravity
    g=9.81,

    # Motor
    Rm=6.7,  # resistance (rated voltage/stall current)

    # back-emf constant (V-s/rad)
    km=0.067,  # (rated voltage / no load speed)

    # Rotary arm
    Mr=0.057,  # mass (kg)
    Lr=0.083,  # length (m)
    Dr=5e-6,   # viscous damping (N-m-s/rad), original: 0.0015

    # Pendulum link
    Mp=0.027,  # mass (kg)
    Lp=0.092,  # length (m)
    Dp=1e-6   # viscous damping (N-m-s/rad), original: 0.0005
)
