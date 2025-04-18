from scipy.signal import butter, lfilter, lfilter_zi


class VelocityFilter:
    def __init__(
        self, order: int, cutoff: float, control_frequency: float, init_vel: float
    ) -> None:
        nyq_freq = control_frequency / 2
        normal_cutoff = cutoff / nyq_freq
        assert normal_cutoff < 1, "cutoff frequency should be smaller than control_frequency / 2"
        self.b, self.a = butter(N=order, Wn=normal_cutoff)
        zi = lfilter_zi(self.b, self.a)
        _, self.z = lfilter(self.b, self.a, x=[init_vel], zi=zi * init_vel)

    def __call__(self, new_vel: float) -> float:
        filtered_vel, self.z = lfilter(self.b, self.a, x=[new_vel], zi=self.z)
        return filtered_vel[0]
